"""Local Hugging Face specialist initialization and numeric-slot fine-tuning."""
import gc
import json
import math
import random
from pathlib import Path
from reactgdiff.pipeline.contracts import discrete_slots, requests, parameter_prompt, parse_proposal, PROMPT_VERSION


def prompt_lengths(tokenizer, prompts, max_length):
    lengths = [len(ids) for ids in tokenizer(prompts, truncation=False).input_ids]
    if lengths and max(lengths) > max_length:
        raise ValueError(f'Parameter prompt exceeds budget: max={max(lengths)}, budget={max_length}, '
                         f'over_budget={sum(n > max_length for n in lengths)}. '
                         'Nothing was truncated. Increase --parameter-max-length.')
    return lengths


def load_skeleton(path, device='cpu'):
    import torch
    from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
    from scripts.train_skeleton_seq2seq import SkeletonSeq2SeqModel, infer_hidden_size
    from reactgdiff.models.graph_codec import GraphTargetCodec
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('checkpoint_type') != 'seq2seq_skeleton': raise ValueError('Expected seq2seq skeleton checkpoint')
    tokenizer = AutoTokenizer.from_pretrained(payload['model_name'], local_files_only=True)
    config = AutoConfig.from_pretrained(payload['model_name'], local_files_only=True)
    base = AutoModelForSeq2SeqLM.from_config(config)
    target_format = payload['config'].get('target_format', 'natural_text')
    if target_format == 'special_tokens':
        tokenizer.add_special_tokens({'additional_special_tokens': list(payload['action_tokens'].values())})
    base.resize_token_embeddings(payload['tokenizer_length'])
    model = SkeletonSeq2SeqModel(base, hidden_size=infer_hidden_size(base),
                                max_steps=payload['config']['max_steps'],
                                length_loss_weight=payload['config'].get('length_loss_weight', 0))
    model.load_state_dict(payload['model_state'], strict=True)
    return model.to(device), tokenizer, GraphTargetCodec.from_dict(payload['codec']), payload


def predict_skeleton(checkpoint, records, device, batch_size=4):
    from scripts.train_skeleton_seq2seq import predict_records
    import torch
    model, tokenizer, codec, payload = load_skeleton(checkpoint, device)
    config = payload['config']
    action_tokens = payload.get('action_tokens', {})
    token_to_action = {int(tokenizer.convert_tokens_to_ids(v)): int(k) for k,v in action_tokens.items()}
    with torch.inference_mode():
        rows = predict_records(model, tokenizer, codec, records,
            target_format=config.get('target_format', 'natural_text'),
            action_token_ids=list(token_to_action), token_to_action_id=token_to_action,
            include_numeric_evidence=config.get('include_numeric_evidence', True),
            prompt_style=config.get('prompt_style', 'compact'),
            max_input_length=config.get('max_input_length', 384),
            max_new_tokens=config.get('max_target_length', 96),
            beam_size=4, num_return_sequences=4, generation_length_penalty=1.0,
            length_prior_weight=config.get('length_prior_weight', 0),
            repetition_penalty_weight=config.get('repetition_penalty_weight', 0),
            batch_size=batch_size, device=device)
    del model, payload
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return rows


def load_parameter_base(model_name, device='cpu'):
    """Load original pretrained weights, independently of the skeleton checkpoint."""
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)
    return model.to(device), tokenizer


def train_parameters(base_model, examples, output, *, epochs=1, batch_size=2, accumulation=8,
                     lr=2e-5, max_length=1024, seed=19, device='cpu', metadata=None, diagnostic_callback=None):
    import torch
    from torch.utils.data import DataLoader
    if not examples: raise ValueError('No numeric training slots')
    output = Path(output)
    if output.exists(): raise FileExistsError(output)
    random.seed(seed); torch.manual_seed(seed)
    print(f'Parameter initialization: original pretrained model {base_model}', flush=True)
    model, tokenizer = load_parameter_base(base_model, device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    lengths = prompt_lengths(tokenizer, [x['prompt'] for x in examples], max_length)
    print(f'Parameter prompts: count={len(lengths)} max_tokens={max(lengths)} budget={max_length}; no truncation', flush=True)
    if diagnostic_callback is not None:
        diagnostic_callback(model, tokenizer, 0, None, 0)
    optimizer_steps = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(examples, batch_size=batch_size, shuffle=True, generator=generator, collate_fn=lambda rows: rows)
    losses = []
    for epoch in range(epochs):
        model.train(); optimizer.zero_grad(); total = 0.0
        for i, rows in enumerate(loader):
            encoded = tokenizer([x['prompt'] for x in rows], padding=True, truncation=False, return_tensors='pt').to(device)
            labels = tokenizer(text_target=[x['target'] for x in rows], padding=True, return_tensors='pt').input_ids.to(device)
            labels[labels == tokenizer.pad_token_id] = -100
            # Correct scaling for the final incomplete accumulation group.
            group_size = min(accumulation, len(loader) - (i//accumulation)*accumulation)
            loss = model(**encoded, labels=labels).loss
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite parameter loss')
            (loss/group_size).backward(); total += float(loss.detach())
            if (i+1) % accumulation == 0 or i+1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); optimizer.zero_grad()
                optimizer_steps += 1
            if i % 20 == 0: print(f'Parameter train epoch={epoch+1} batch={i+1}/{len(loader)} loss={float(loss):.4f}', flush=True)
        losses.append(total/len(loader))
        if diagnostic_callback is not None:
            diagnostic_callback(model, tokenizer, epoch+1, losses[-1], optimizer_steps)
    model.config.use_cache = True
    model.save_pretrained(output); tokenizer.save_pretrained(output)
    info = {'initialization': str(base_model), 'initialization_kind': 'original_pretrained', 'examples': len(examples), 'epochs': epochs,
            'train_loss': losses, 'optimizer_steps': optimizer_steps, 'learning_rate': lr, 'batch_size': batch_size, 'accumulation': accumulation, 'seed': seed, 'max_length': max_length,
            'prompt_version': PROMPT_VERSION, 'prompt_max_tokens': max(lengths),
            'prompt_mean_tokens': sum(lengths)/len(lengths),
            'selection': 'fixed_epochs_no_validation_selection', **(metadata or {})}
    (output/'parameter_training.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    del model, optimizer
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return info


class ParameterGenerator:
    def __init__(self, path, device='cpu', batch_size=4, max_length=1024):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path, local_files_only=True).to(device).eval()
        self.device, self.batch_size, self.max_length = device, batch_size, max_length
        self.truncated_prompts = 0
        self.prompt_count = 0
        self.prompt_max_tokens = 0

    def generate(self, record, slots, include_source=False):
        import torch
        prompts = [parameter_prompt(record, slots, request, include_source) for request in requests(slots)]
        raw = []
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i:i+self.batch_size]
            lengths = prompt_lengths(self.tokenizer, batch, self.max_length)
            self.prompt_count += len(lengths)
            self.prompt_max_tokens = max(self.prompt_max_tokens, max(lengths, default=0))
            encoded = self.tokenizer(batch, padding=True, truncation=False, return_tensors='pt').to(self.device)
            with torch.inference_mode():
                result = self.model.generate(**encoded, max_new_tokens=24, num_beams=1, do_sample=False)
            raw.extend(self.tokenizer.batch_decode(result, skip_special_tokens=True))
        return [parse_proposal(x) for x in raw], raw
