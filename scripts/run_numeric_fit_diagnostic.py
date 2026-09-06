"""Training-fit diagnostic only; gold graphs and source-assisted arm are not benchmark results."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import gc
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.run_fixed_pipeline import indexed, digest, save
from reactgdiff.utils.io import read_jsonl, write_jsonl
from reactgdiff.models.graph_codec import GraphTargetCodec
from reactgdiff.pipeline.contracts import discrete_slots, requests, parameter_prompt, parse_proposal, input_record, PROMPT_VERSION
from reactgdiff.data.numeric_evidence import numeric_candidates_from_record, normalize_unit
from reactgdiff.pipeline.specialist import train_parameters, prompt_lengths


def examples_from_records(records, codec, include_source):
    examples = []
    for record in records:
        gold = codec.target_slots_from_record(record)
        slots = discrete_slots(gold)
        evidence = numeric_candidates_from_record(input_record(record, include_source), include_source=include_source)
        for request in requests(slots):
            value = gold[request['step']]['quantity_slots'][request['quantity']].get('value')
            if value is None or not math.isfinite(value):
                continue
            examples.append({'index': record['index'], 'request': request,
                'prompt': parameter_prompt(record, slots, request, include_source),
                'target': f'{value:.8g}',
                'matching_value_unit_in_input': any(c.value is not None and
                    normalize_unit(c.unit or '') == normalize_unit(request['unit']) and
                    math.isclose(c.value, value, rel_tol=1e-6, abs_tol=1e-8) for c in evidence)})
    return examples


def summarize_predictions(rows):
    valid = [r for r in rows if r['value'] is not None]
    exact = sum(r['value'] is not None and math.isclose(r['value'], float(r['target']), rel_tol=1e-6, abs_tol=1e-8) for r in rows)
    within = sum(r['value'] is not None and abs(r['value']-float(r['target'])) <= max(1e-8, .1*abs(float(r['target']))) for r in rows)
    return {'count': len(rows), 'valid_numeric_rate': len(valid)/max(1,len(rows)),
            'numeric_exact_rate': exact/max(1,len(rows)), 'within_10_percent_rate': within/max(1,len(rows)),
            'zero_outputs': sum(r['value'] == 0 for r in valid),
            'raw_output_histogram': dict(Counter(r['raw'] for r in rows))}


def evaluate(model, tokenizer, examples, batch_size, max_length, device):
    import torch
    model.eval()
    rows = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start+batch_size]
        prompt_lengths(tokenizer, [r['prompt'] for r in batch], max_length)
        encoded = tokenizer([r['prompt'] for r in batch], padding=True, truncation=False, return_tensors='pt').to(device)
        with torch.inference_mode():
            generated = model.generate(**encoded, max_new_tokens=24, num_beams=1, do_sample=False)
        for example, raw in zip(batch, tokenizer.batch_decode(generated, skip_special_tokens=True)):
            rows.append({k:v for k,v in example.items() if k != 'prompt'} | {'raw':raw, 'value':parse_proposal(raw)})
    metrics = summarize_predictions(rows)
    metrics['by_input_evidence_match'] = {str(flag): summarize_predictions([r for r in rows if r['matching_value_unit_in_input'] == flag]) for flag in (True,False)}
    return metrics, rows


def main():
    import torch
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', default='outputs/prepared_splits/openexp/scale_small/train.jsonl')
    p.add_argument('--validation', default='outputs/prepared_splits/openexp/scale_small/val.jsonl')
    p.add_argument('--graph-checkpoint', default='outputs/checkpoints/openexp_small_hash_pointer_v2.pt')
    p.add_argument('--base-model', default='/home/void/models/molt5-base')
    p.add_argument('--train-records', type=int, default=32)
    p.add_argument('--validation-records', type=int, default=32)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--accumulation', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--max-length', type=int, default=4096)
    p.add_argument('--seed', type=int, default=19)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output')
    p.add_argument('--overlength-policy', choices=['error','skip'], default='error')
    p.add_argument('--train-eval-records', type=int, default=32)
    p.add_argument('--save-every-epoch', action='store_true')
    p.add_argument('--warm-start-run', help='Initialize from a previous numeric model, with new data and fresh optimizer')
    p.add_argument('--arm', choices=['both','free','assisted'], default='both')
    p.add_argument('--resume-run', help='Previous diagnostic result directory; resume weights with a fresh optimizer')
    args = p.parse_args()
    previous = None
    if args.resume_run and args.warm_start_run:
        p.error('Choose resume-run or warm-start-run, not both')
    if args.resume_run:
        previous = json.loads((Path(args.resume_run)/'report.json').read_text())
        if any(a.get('training',{}).get('prompt_version') != PROMPT_VERSION for a in previous['arms'].values()):
            p.error('Prompt version changed: use --warm-start-run for a new experiment')
        # Preserve the original experiment and samples. Only duration, arm and output change.
        for key, value in previous['config'].items():
            if key not in ('epochs','arm','resume_run','output'):
                setattr(args, key, value)
    if args.train_records < 0 or min(args.train_eval_records,args.validation_records,args.epochs,args.batch_size,args.accumulation,args.max_length) < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        p.error('Counts and learning rate must be positive and finite')
    train, val = list(read_jsonl(args.train)), list(read_jsonl(args.validation))
    train_index, val_index = indexed(train), indexed(val)
    if set(train_index) & set(val_index):
        raise ValueError('Training and validation IDs overlap')
    for split, records in [('train',train),('val',val)]:
        if any(r.get('_split') != split for r in records):
            raise ValueError('Expected explicit '+split+' split; test data is not allowed')
    if args.train_records > len(train) or args.validation_records > len(val):
        raise ValueError('Requested more records than available')
    chosen_train = random.Random(args.seed).sample(train, args.train_records or len(train))
    chosen_val = random.Random(args.seed+1).sample(val, args.validation_records)
    payload = torch.load(args.graph_checkpoint, map_location='cpu', weights_only=False)
    codec = GraphTargetCodec.from_dict(payload['codec'])
    del payload
    output = Path(args.output or ('outputs/numeric_fit/'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    output.mkdir(parents=True, exist_ok=False)
    report = {'config': vars(args), 'purpose':'diagnostic_only_gold_graphs', 'prompt_version':PROMPT_VERSION,
        'selection':'fixed_epochs_no_validation_selection',
        'limitations':['Source-assisted input may contain target procedure; never compare it as source-free performance.',
                      'Input value/unit match does not establish correct material or step binding.',
                      'Training fit is not generalization. Gold graphs remove upstream prediction errors.',
                      'Train metrics refer only to the fixed training probe, not the full training set.',
                      'Gold material bindings are not predicted by legacy diffusion checkpoints.'],
        'train_sha256':digest(args.train), 'validation_sha256':digest(args.validation),
        'train_ids':[r['index'] for r in chosen_train], 'validation_ids':[r['index'] for r in chosen_val], 'arms':{}}
    if previous:
        for key in ('train_sha256','validation_sha256','train_ids','validation_ids'):
            if report[key] != previous[key]:
                raise ValueError('Resume data mismatch: '+key)
        report['resume_from'] = str(Path(args.resume_run).resolve())
        report['optimizer_resume'] = 'fresh_optimizer_previous_state_not_saved'
    save(output/'report.json', report)
    for arm, include_source in [('source_free',False),('source_assisted_DIAGNOSTIC',True)]:
        if args.arm == 'free' and include_source or args.arm == 'assisted' and not include_source:
            continue
        initial_model = args.base_model
        prior_epochs = 0
        if previous:
            if arm not in previous['arms']:
                raise ValueError('Requested arm absent from previous run: '+arm)
            prior_epochs = previous['arms'][arm]['curve'][-1]['epoch']
            initial_model = str(Path(args.resume_run)/arm/'model')
            if not Path(initial_model).is_dir():
                raise FileNotFoundError(initial_model)
            print(f'Resuming {arm} from epoch {prior_epochs}; fresh optimizer', flush=True)
        if args.warm_start_run:
            warm = json.loads((Path(args.warm_start_run)/'report.json').read_text())
            if set(map(str,warm.get('train_ids',[]))) & set(val_index):
                raise ValueError('Warm-start model has validation training exposure')
            if arm not in warm.get('arms', {}):
                raise ValueError('Warm-start arm absent: '+arm)
            initial_model = str(Path(args.warm_start_run)/arm/'model')
            if not Path(initial_model).is_dir():
                raise FileNotFoundError(initial_model)
        arm_dir = output/arm
        arm_dir.mkdir()
        training = examples_from_records(chosen_train, codec, include_source)
        validation = examples_from_records(chosen_val, codec, include_source)
        length_audit = {}
        if args.overlength_policy == 'skip':
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(initial_model,local_files_only=True)
            def filter_length(examples, split):
                kept, dropped = [], []
                for start in range(0,len(examples),256):
                    batch = examples[start:start+256]
                    ids = tokenizer([e['prompt'] for e in batch],truncation=False).input_ids
                    for e,tokens in zip(batch,ids):
                        if len(tokens) <= args.max_length:
                            kept.append(e)
                        else:
                            dropped.append({'index':e['index'],'request':e['request'],'tokens':len(tokens)})
                    if start % 10240 == 0: print(f'{arm} preflight {split} {start}/{len(examples)}',flush=True)
                write_jsonl(arm_dir/f'{split}_overlength_exclusions.jsonl',dropped)
                length_audit[split] = {'before':len(examples),'retained':len(kept),'excluded':len(dropped)}
                return kept
            training = filter_length(training,'train')
            validation = filter_length(validation,'validation')
            del tokenizer
        probe_ids = {r['index'] for r in chosen_train[:args.train_eval_records]}
        training_probe = [e for e in training if e['index'] in probe_ids]
        if not training or not validation or not training_probe:
            raise ValueError('No numeric slots for diagnostic')
        if include_source and not any(str(r.get('source','')).strip() for r in chosen_train):
            raise ValueError('Source-assisted diagnostic requires actual source text')
        write_jsonl(arm_dir/'training_examples.jsonl',training)
        write_jsonl(arm_dir/'validation_examples.jsonl',validation)
        targets_by_prompt = defaultdict(set)
        for e in training:
            targets_by_prompt[e['prompt']].add(e['target'])
        binding_counts = Counter(json.loads(e['prompt'].split('\n',1)[1])['requested_context']['binding'] for e in training)
        entry = {'length_audit':length_audit, 'train_probe_examples':len(training_probe), 'train_probe_ids':list(probe_ids),
                 'material_binding_counts':dict(binding_counts),
                 'training_records_selected':len(chosen_train),
                 'training_records_with_numeric_slots':len({e['index'] for e in training}),
                 'codec_capacity':{'max_steps':codec.max_steps,'max_material_slots':codec.max_material_slots},
                 'train_examples':len(training), 'validation_examples':len(validation),
                 'conflicting_training_prompts':sum(len(v)>1 for v in targets_by_prompt.values()), 'curve':[]}
        report['arms'][arm] = entry
        def callback(model, tokenizer, epoch, loss, steps):
            metrics, rows = evaluate(model,tokenizer,training_probe,args.batch_size,args.max_length,args.device)
            point = {'epoch':prior_epochs+epoch, 'additional_epoch':epoch, 'optimizer_steps':steps, 'train_loss':loss, 'train':metrics}
            write_jsonl(arm_dir/f'train_epoch_{prior_epochs+epoch:03d}.jsonl',rows)
            if epoch in (0,args.epochs):
                val_metrics, val_rows = evaluate(model,tokenizer,validation,args.batch_size,args.max_length,args.device)
                point['validation'] = val_metrics
                write_jsonl(arm_dir/f'validation_epoch_{prior_epochs+epoch:03d}.jsonl',val_rows)
            if epoch and args.save_every_epoch:
                snapshot = arm_dir/f'checkpoint_epoch_{prior_epochs+epoch:03d}'
                model.save_pretrained(snapshot)
                tokenizer.save_pretrained(snapshot)
                save(snapshot/'diagnostic_provenance.json', {'prompt_version':PROMPT_VERSION,
                    'epoch':prior_epochs+epoch,'train_sha256':report['train_sha256'],
                    'include_source':include_source,'optimizer_state_saved':False})
            entry['curve'].append(point)
            save(output/'report.json',report)
            print(f'{arm} epoch={prior_epochs+epoch} steps={steps} train_valid={metrics["valid_numeric_rate"]:.3f} train_exact={metrics["numeric_exact_rate"]:.3f}', flush=True)
        entry['training'] = train_parameters(initial_model,training,arm_dir/'model',epochs=args.epochs,
            batch_size=args.batch_size,accumulation=args.accumulation,lr=args.lr,max_length=args.max_length,
            seed=args.seed,device=args.device,metadata={'diagnostic_only':True,'include_source':include_source,'prior_epochs':prior_epochs,
                'initialization_kind':'continued_diagnostic_weights' if (previous or args.warm_start_run) else 'original_pretrained',
                'train_ids':report['train_ids'],'train_file_sha256':report['train_sha256'],'graph_input':'gold_ORACLE'},diagnostic_callback=callback)
        save(output/'report.json', report)
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    lines = ['# Numeric fit diagnostic (not benchmark performance)', '', '| Arm | Train valid | Train exact | Validation valid | Validation exact |', '|---|---:|---:|---:|---:|']
    for arm,entry in report['arms'].items():
        end = entry['curve'][-1]
        lines.append(f'| {arm} | {end["train"]["valid_numeric_rate"]:.4f} | {end["train"]["numeric_exact_rate"]:.4f} | {end["validation"]["valid_numeric_rate"]:.4f} | {end["validation"]["numeric_exact_rate"]:.4f} |')
    (output/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'Results: {output}',flush=True)


if __name__ == '__main__':
    main()
