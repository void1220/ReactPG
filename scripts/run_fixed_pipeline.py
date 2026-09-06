"""Run fixed-skeleton continuous filling plus controlled bottleneck diagnostics.

Default uses existing graph/skeleton checkpoints, trains a small numeric filler,
and evaluates validation only. Gold interventions are diagnostics, never deployed
predictions. Operation completion is disabled and enabling it fails explicitly.
"""
import argparse
from collections import Counter
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from reactgdiff.data.action_parser import parse_action_sequence
from reactgdiff.eval.structured import corpus_structured_metrics, anchors, edits, VERSION
from reactgdiff.eval.semantic import corpus_semantic_metrics
from reactgdiff.eval.text import corpus_text_metrics
from reactgdiff.pipeline.contracts import input_record, discrete_slots, requests, parameter_prompt, fill_values, validate, PROMPT_VERSION
from reactgdiff.utils.io import read_jsonl, write_jsonl


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def indexed(rows):
    result = {}
    for row in rows:
        key = str(row.get('index'))
        if key == 'None' or key in result: raise ValueError('Missing or duplicate record index: '+key)
        result[key] = row
    return result


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def evaluate(rows):
    pairs = [(row['predicted_actions'], row['reference_actions']) for row in rows]
    return {**corpus_text_metrics(pairs), **corpus_semantic_metrics(pairs), **corpus_structured_metrics(pairs)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', default='outputs/prepared_splits/openexp/scale_small/val.jsonl')
    p.add_argument('--train', default='outputs/prepared_splits/openexp/scale_small/train.jsonl')
    p.add_argument('--graph-checkpoint', default='outputs/checkpoints/openexp_small_hash_pointer_v2.pt')
    p.add_argument('--skeleton-checkpoint', default='outputs/checkpoints/openexp_small_hash_v1_seq2seq_skeleton.pt')
    p.add_argument('--skeleton-cache', default='outputs/skeleton/openexp_small_hash_v1_seq2seq_skeleton_val.jsonl')
    p.add_argument('--regenerate-skeleton', action='store_true')
    p.add_argument('--parameter-model', help='Existing HF numeric model directory; otherwise train in output directory')
    p.add_argument('--parameter-train-records', type=int, default=2048)
    p.add_argument('--parameter-max-length', type=int, default=1024)
    p.add_argument('--parameter-epochs', type=int, default=1)
    p.add_argument('--limit', type=int, default=128)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--sample-steps', type=int, default=32)
    p.add_argument('--quantity-threshold', type=float, default=.999)
    p.add_argument('--include-source', action='store_true', help='Explicit source-assisted protocol; source may reveal target content. Never compare to source-free baselines.')
    p.add_argument('--allow-operation-completion', action='store_true', help='Reserved; not implemented in this diagnostic phase')
    p.add_argument('--rules', help='Optional JSON list of sourced operation/unit bounds; no chemical bounds are invented')
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=19)
    p.add_argument('--output', default=None)
    args = p.parse_args()
    if args.allow_operation_completion: p.error('Operation completion is disabled in this release')
    if min(args.limit, args.parameter_train_records, args.parameter_epochs, args.batch_size, args.sample_steps, args.parameter_max_length) <= 0: p.error('Counts must be positive')
    if not 0 <= args.quantity_threshold <= 1: p.error('Threshold must be in [0,1]')
    import torch
    from reactgdiff.models.procedure_graph_diffusion import load_procedure_graph_diffusion_checkpoint, predict_procedure_graph_diffusion_records
    from reactgdiff.models.joint_diffusion import ReactGDiffFeaturizer
    from reactgdiff.pipeline.specialist import predict_skeleton, train_parameters, ParameterGenerator
    torch.manual_seed(args.seed); random.seed(args.seed)
    out = Path(args.output or ('outputs/fixed_pipeline/' + time.strftime('%Y%m%d_%H%M%S')))
    out.mkdir(parents=True, exist_ok=False)
    rules = json.loads(Path(args.rules).read_text(encoding='utf-8')) if args.rules else []
    validate({}, [], rules=rules)  # validate rule configuration before any model work
    start = time.monotonic()
    report = {'protocol': 'source_assisted' if args.include_source else 'source_free_with_condition_maps',
              'allow_operation_completion': False, 'metric_version': VERSION, 'config': vars(args),
              'stages': {}, 'timings': {}, 'supplied_rules': rules, 'limitations': [
                  'Existing graph denoiser retained; legacy numeric heads remain internal but new filler never reads numeric predictions.',
                  'Temperature/duration references are resolved from input maps, not regenerated.',
                  'Gate is physical/reference validity, not chemical safety verification.',
                  'Oracle interventions diagnose headroom, not additive causal attribution.',
                  'Semantic-A is an OpenExp adaptation; monotone-anchor Tau is degenerate.',
                  'Current graph binds quantities to operations, not reliably to individual materials.']}
    save(out/'run_config.json', report)
    try:
        report['commit'] = subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()
    except Exception: report['commit'] = 'unavailable'
    # Audit whole split IDs (not just the evaluation prefix) before training.
    print('[1/6] Checking data split IDs and fingerprints', flush=True)
    eval_rows = list(read_jsonl(args.input)); train_rows = list(read_jsonl(args.train))
    eval_by_id, train_by_id = indexed(eval_rows), indexed(train_rows)
    overlap = set(eval_by_id) & set(train_by_id)
    if overlap: raise ValueError(f'Train/eval ID overlap: {len(overlap)}')
    for row in train_rows:
        if row.get('_split') not in (None, 'train'): raise ValueError('Nontraining record in train file')
    for row in eval_rows:
        if row.get('_split') not in (None, 'val', 'validation', 'dev'): raise ValueError('This diagnostic runner accepts validation only')
    report['data'] = {'train_total': len(train_rows), 'validation_total': len(eval_rows),
                      'train_sha256': digest(args.train), 'validation_sha256': digest(args.input), 'id_overlap': 0}
    random.Random(args.seed).shuffle(eval_rows)
    records = eval_rows[:args.limit]
    if not records: raise ValueError('Empty validation set')
    safe = [input_record(r, args.include_source) for r in records]
    report['data']['evaluation_ids'] = [r['index'] for r in records]
    print('[2/6] Loading or generating fixed MolT5 skeletons', flush=True)
    tick = time.monotonic()
    if not args.regenerate_skeleton and Path(args.skeleton_cache).is_file():
        cache = indexed(list(read_jsonl(args.skeleton_cache)))
        if any(str(r['index']) not in cache for r in records): raise ValueError('Incomplete skeleton cache; use --regenerate-skeleton')
        report['skeleton_cache_sha256'] = digest(args.skeleton_cache)
        report['limitations'].append('Cached skeleton provenance is historical; use --regenerate-skeleton to enforce current input policy.')
    else:
        cache = indexed(predict_skeleton(args.skeleton_checkpoint, safe, args.device, args.batch_size))
    predicted_ops = {}
    from reactgdiff.models.procedure_graph_diffusion import _operations_from_skeleton_payload
    confusion = Counter(); skeleton_diagnostics = []
    for r in records:
        key = str(r['index']); pred = _operations_from_skeleton_payload(cache[key])
        gold = [s.operation_type for s in parse_action_sequence(r['actions'])]
        error, conf = edits(pred, gold); confusion.update(conf)
        candidates = cache[key].get('topk_skeletons') or []
        oracle_topk = any(c.get('operations') == gold for c in candidates if isinstance(c, dict)) if candidates else None
        skeleton_diagnostics.append({'index': r['index'], 'predicted': pred, 'reference': gold,
                                     'edits': error, 'topk_exact_oracle': oracle_topk,
                                     'raw_model_text': cache[key].get('predicted_skeleton_text'),
                                     'same_operation_multiset_wrong_order': pred != gold and Counter(pred) == Counter(gold)})
        predicted_ops[key] = {'operations': pred}
    write_jsonl(out/'skeleton_diagnostics.jsonl', skeleton_diagnostics)
    report['skeleton_confusions'] = dict(confusion)
    report['timings']['skeleton_seconds'] = time.monotonic()-tick
    print('[3/6] Sampling graph with predicted and gold skeletons', flush=True)
    model, codec, features, _ = load_procedure_graph_diffusion_checkpoint(args.graph_checkpoint, device=args.device)
    featurizer = ReactGDiffFeaturizer.from_dict(features)
    vectors = [featurizer.condition_vector(r) for r in safe]
    report['graph_checkpoint_config'] = model.config()
    report['graph_condition_featurizer'] = features
    report['checkpoints'] = {'graph_sha256': digest(args.graph_checkpoint), 'skeleton_sha256': digest(args.skeleton_checkpoint)}
    gold_cache = {str(r['index']): {'operations': [s.operation_type for s in parse_action_sequence(r['actions'])]} for r in records}
    report['data']['reference_over_capacity'] = sum(len(v['operations']) > codec.max_steps-1 for v in gold_cache.values())

    def sample(cache, legacy=False):
        valid = [i for i,r in enumerate(records) if 0 < len(cache[str(r['index'])]['operations']) <= codec.max_steps-1
                 and all(op in codec.action_vocab for op in cache[str(r['index'])]['operations'])]
        predictions = []
        for start_i in range(0, len(valid), args.batch_size):
            ids = valid[start_i:start_i+args.batch_size]
            with torch.inference_mode():
                predictions += predict_procedure_graph_diffusion_records(model, codec, [safe[i] for i in ids],
                    condition_vectors=[vectors[i] for i in ids], skeleton_source='cache', skeleton_cache=cache,
                    quantity_gate_threshold=args.quantity_threshold, condition_probability_threshold=.05,
                    decode_quantity_values=False, ground_numeric_slots=legacy,
                    drop_unsupported_numeric_slots=legacy, numeric_candidate_reuse_penalty=16 if legacy else 0,
                    numeric_candidate_unit_weight=1 if legacy else 0, sample_steps=args.sample_steps,
                    sample_mode='sample_argmax_final', sampler='posterior', sample_batch_size=args.batch_size,
                    seed=args.seed+start_i, device=args.device)
            print(f'Graph {"legacy" if legacy else "slots"}: {min(start_i+len(ids),len(valid))}/{len(valid)}', flush=True)
        found = indexed(predictions)
        result = []
        for r in records:
            key = str(r['index'])
            row = found.get(key, {'index': r['index'], 'decoded_slots': [], 'predicted_actions': '', 'pipeline_error': 'empty_unknown_or_over_capacity_skeleton'})
            actual = [s['operation_type'] for s in row['decoded_slots']]
            if actual != cache[key]['operations'] and key in found:
                row = {'index': r['index'], 'decoded_slots': [], 'predicted_actions': '', 'pipeline_error': 'decoder_changed_fixed_skeleton', 'requested_operations': cache[key]['operations'], 'decoded_operations': actual}
            row['reference_actions'] = r['actions']; result.append(row)
        return result

    tick = time.monotonic()
    legacy = sample(predicted_ops, legacy=True)
    pred = sample(predicted_ops)
    gold_skeleton = sample(gold_cache)
    report['timings']['graph_seconds'] = time.monotonic()-tick
    # Save raw slots before training; failures still leave diagnosable artifacts.
    write_jsonl(out/'legacy_pointer.jsonl', legacy)
    write_jsonl(out/'discrete_predicted_skeleton.jsonl', pred)
    write_jsonl(out/'discrete_gold_skeleton_ORACLE.jsonl', gold_skeleton)
    del model
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    report['stages']['legacy_pointer'] = evaluate(legacy)
    save(out/'partial_report.json',report)
    report['graph_errors'] = {name: dict(Counter(r.get('pipeline_error') for r in rows if r.get('pipeline_error'))) for name,rows in [('predicted',pred),('gold_skeleton',gold_skeleton)]}
    print('[4/6] Training/loading independent numeric specialist', flush=True)
    tick = time.monotonic()
    parameter_path = args.parameter_model or str(out/'parameter_model')
    if not args.parameter_model:
        random.Random(args.seed).shuffle(train_rows)
        subset = train_rows[:args.parameter_train_records]
        examples = []
        for r in subset:
            target = codec.target_slots_from_record(r)
            slots = discrete_slots(target)
            for request in requests(slots):
                value = target[request['step']]['quantity_slots'][request['quantity']].get('value')
                if value is not None:
                    examples.append({'prompt': parameter_prompt(r, slots, request, args.include_source), 'target': f'{value:.8g}'})
        report['parameter_training'] = train_parameters(args.skeleton_checkpoint, examples, parameter_path,
            epochs=args.parameter_epochs, batch_size=min(args.batch_size,2), device=args.device, seed=args.seed, max_length=args.parameter_max_length,
            metadata={'include_source': args.include_source, 'train_file_sha256': report['data']['train_sha256'], 'train_ids': [r['index'] for r in subset]})
    else:
        metadata_path = Path(parameter_path)/'parameter_training.json'
        if not metadata_path.is_file(): raise ValueError('Parameter model lacks training provenance')
        meta = json.loads(metadata_path.read_text())
        if meta.get('prompt_version') != PROMPT_VERSION: raise ValueError('Parameter prompt version changed: train a new filler; do not reuse v1 checkpoint')
        if bool(meta.get('include_source')) != args.include_source: raise ValueError('Parameter model input policy mismatch')
        if set(map(str, meta.get('train_ids', []))) & set(eval_by_id): raise ValueError('Parameter model trained on validation IDs')
        report['parameter_training'] = meta
    report['timings']['parameter_training_seconds'] = time.monotonic()-tick
    filler = ParameterGenerator(parameter_path, args.device, args.batch_size, args.parameter_max_length)
    print('[5/6] Filling, gating, compiling and running oracle controls', flush=True)
    panels = {k: [] for k in ('new_before_gate','new_after_gate','gold_skeleton_ORACLE','gold_discrete_ORACLE','gold_numeric_ORACLE','compiler_structured_ORACLE','compiler_surface_ORACLE')}
    gate_rows = []
    tick = time.monotonic()
    for i,(r, predicted, gs) in enumerate(zip(records,pred,gold_skeleton)):
        gold = codec.target_slots_from_record(r)
        def compile_slots(slots): return codec.decompile_generated_graph(codec.build_generated_graph(input_record(r,args.include_source), slots))
        def row(text, **extra): return {'index': r['index'], 'predicted_actions': text, 'reference_actions': r['actions'], **extra}
        base = discrete_slots(predicted['decoded_slots'])
        values, raw = filler.generate(r, base, args.include_source)
        filled = fill_values(base, values)
        gate = validate(input_record(r,args.include_source), filled, args.include_source, rules=rules)
        graph = codec.build_generated_graph(input_record(r,args.include_source), filled)
        graph['metadata']['validity_gate'] = gate
        graph['metadata']['chemical_safety_verified'] = False
        text = codec.decompile_generated_graph(graph) if filled else ''
        panels['new_before_gate'].append(row(text, decoded_slots=filled, parameter_raw=raw, gate=gate, graph=graph))
        panels['new_after_gate'].append(row(text if gate['status']=='PASS' else '', gate=gate))
        gate_rows.append(gate)
        for name, structure in [('gold_skeleton_ORACLE',gs['decoded_slots']),('gold_discrete_ORACLE',gold)]:
            slots = discrete_slots(structure)
            values, raw = filler.generate(r,slots,args.include_source)
            panels[name].append(row(compile_slots(fill_values(slots,values)) if slots else '', oracle=True))
        # Oracle numeric values only where operation anchors and quantity units align.
        oracle = deepcopy(filled); replaced = 0
        for pi,gi in anchors([s['operation_type'] for s in filled],[s['operation_type'] for s in gold]):
            for j,q in enumerate(oracle[pi]['quantity_slots']):
                gq = gold[gi]['quantity_slots']
                if j < len(gq) and q['unit'] == gq[j]['unit']:
                    q.update(value=gq[j]['value'],text=gq[j]['text'],source='ORACLE'); replaced += 1
        panels['gold_numeric_ORACLE'].append(row(compile_slots(oracle) if oracle else '', oracle=True, replaced_slots=replaced))
        structural_gold = deepcopy(gold)
        for s in structural_gold:
            s.pop('argument_text',None); s.pop('raw_text',None)
        panels['compiler_structured_ORACLE'].append(row(compile_slots(structural_gold),oracle=True))
        panels['compiler_surface_ORACLE'].append(row(compile_slots(gold),oracle=True))
        for name,rows in panels.items():
            # Append completed record immediately; interrupted runs retain progress.
            with (out/(name+'.jsonl')).open('a',encoding='utf-8') as f: f.write(json.dumps(rows[-1],ensure_ascii=False)+'\n')
        print(f'Pipeline {i+1}/{len(records)} gate={gate["status"]}',flush=True)
    report['timings']['fill_diagnostics_seconds'] = time.monotonic()-tick
    for name,rows in panels.items(): report['stages'][name] = evaluate(rows)
    report['compiled_surface_consistency'] = sum(
        [step.operation_type for step in parse_action_sequence(row['predicted_actions'])] == [s['operation_type'] for s in row['decoded_slots']]
        for row in panels['new_before_gate'])/len(records)
    passed = [row for row in panels['new_before_gate'] if row['gate']['status']=='PASS']
    report['gate'] = {'pass_rate':len(passed)/len(records), 'rejected':len(records)-len(passed),
                     'accepted_only_metrics':evaluate(passed),
                     'issues':dict(Counter(x['rule'] for g in gate_rows for x in g['issues'])),
                     'unsupported_parameters':sum(g['unsupported_parameter_count'] for g in gate_rows),
                     'total_parameters':sum(g['parameter_count'] for g in gate_rows)}
    exact_ids = {str(x['index']) for x in skeleton_diagnostics if x['predicted'] == x['reference']}
    report['by_skeleton_correctness'] = {name: evaluate([r for r in panels['new_before_gate'] if (str(r['index']) in exact_ids) == flag]) for name,flag in [('exact',True),('incorrect',False)]}
    base_metrics = report['stages']['new_before_gate']
    report['oracle_headroom_deltas'] = {name: {k: report['stages'][name][k]-base_metrics[k] for k in ('semantic_score','levenshtein_75_rate','aligned_parameter_iou')} for name in ('gold_skeleton_ORACLE','gold_discrete_ORACLE','gold_numeric_ORACLE')}
    report['parameter_prompt_truncations'] = filler.truncated_prompts
    report['parameter_prompts'] = {'version': PROMPT_VERSION, 'count': filler.prompt_count, 'max_tokens': filler.prompt_max_tokens}
    quantities = [q for row in panels['new_before_gate'] for step in row['decoded_slots'] for q in step['quantity_slots']]
    report['parameter_generation'] = {
        'predicted_slots': len(quantities),
        'reference_quantity_count': sum(len(s.quantities) for r in records for s in parse_action_sequence(r['actions'])),
        'valid_numeric_outputs': sum(q.get('value') is not None for q in quantities),
        'zero_outputs': sum(q.get('value') == 0 for q in quantities),
        'raw_output_histogram': dict(Counter(x for row in panels['new_before_gate'] for x in row['parameter_raw'])),
        'predicted_unit_histogram': dict(Counter(q['unit'] for q in quantities))}
    if torch.cuda.is_available(): report['gpu_peak_allocated_mib'] = torch.cuda.max_memory_allocated()/1024**2
    report['timings']['total_seconds'] = time.monotonic()-start
    save(out/'report.json',report)
    text = ['# Fixed-skeleton diagnostic report','', '| Stage | LEV75 | Semantic | Step-M | Order-LCS | Aligned parameters |','|---|---:|---:|---:|---:|---:|']
    for name,m in report['stages'].items(): text.append(f'| {name} | {m["levenshtein_75_rate"]:.4f} | {m["semantic_score"]:.4f} | {m["thoth_step_m"]:.4f} | {m["thoth_order_lcs"]:.4f} | {m["aligned_parameter_iou"]:.4f} |')
    text += ['', 'ORACLE rows use references for diagnosis only. Differences are not additive causal contributions.', '', *report['limitations']]
    (out/'summary.md').write_text('\n'.join(text),encoding='utf-8')
    print('[6/6] Done. Return '+str(out/'report.json')+' and skeleton_diagnostics.jsonl',flush=True)


if __name__ == '__main__': main()
