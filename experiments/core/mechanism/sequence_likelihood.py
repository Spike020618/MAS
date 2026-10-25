"""Explicit teacher-forced sequence likelihoods; no sampled-code proxy."""
from math import fsum, isfinite


def checkpoint_digest(model_path):
    """Content commitment of all checkpoint files, including symlink targets."""
    from pathlib import Path
    from hashlib import sha256
    root = Path(model_path).resolve()
    files = sorted(p for p in root.rglob('*') if p.is_file())
    if not files:
        raise ValueError('empty checkpoint')
    manifest = sha256()
    for path in files:
        # Hugging Face cache symlinks are permitted, but bytes are hashed.
        content = sha256()
        with path.open('rb') as stream:
            while chunk := stream.read(1024*1024): content.update(chunk)
        manifest.update(str(path.relative_to(root)).encode()+b'\0'+content.digest())
    return manifest.hexdigest()


def sum_token_logprobs(values):
    values = tuple(values)
    if not values or any(type(x) not in (float, int) or not isfinite(x) or x > 0 for x in values):
        raise ValueError('complete finite nonpositive target token logprobs required')
    return fsum(values)


def requests_for_case(case):
    """Plain-text, documented reference prompts; not task-original templates.

    Choice strings are full answers, not choice IDs. IDs only index records.
    Caller must not supply labels or extra fields to the scoring stage.
    """
    if set(case) != {'id', 'question', 'candidates', 'background_correction'}:
        raise ValueError('closed public case schema; labels and extra fields forbidden')
    if not isinstance(case['id'], str) or not case['id'] or not isinstance(case['question'], str) or not case['question']:
        raise ValueError('nonempty question and ID required')
    if type(case['background_correction']) is not bool:
        raise ValueError('explicit background correction flag required')
    candidates = case['candidates']
    if not isinstance(candidates, dict) or len(candidates) < 2:
        raise ValueError('at least two candidate ID/text pairs required')
    if any(not isinstance(k, str) or not k or not isinstance(v, str) or not v for k, v in candidates.items()):
        raise ValueError('nonempty text candidates required')
    if len(set(candidates.values())) != len(candidates):
        raise ValueError('duplicate candidate text')
    jobs = {}
    for cid, text in candidates.items():
        for kind, ending in (('g_positive', 'Answer:'), ('g_negative', 'Incorrect Answer:')):
            jobs[f'{kind}/{cid}'] = {'context': f"Question: {case['question']}\n{ending}", 'target': ' ' + text}
        if case['background_correction']:
            jobs[f'background/{cid}'] = {'context': 'Answer:', 'target': ' ' + text}
        for verdict in ('correct', 'incorrect'):
            jobs[f'd/{cid}/{verdict}'] = {
                'context': f"Question: {case['question']}\nProposed answer: {text}\nThe proposed answer is",
                'target': ' ' + verdict}
            if case['background_correction']:
                jobs[f'd_background/{verdict}'] = {'context': 'Answer:', 'target': ' ' + verdict}
    return jobs


def score_tables(case, scored):
    expected = requests_for_case(case)
    if set(scored) != set(expected):
        raise ValueError('missing or unexpected sequence score slots')
    values = {}
    for key, req in expected.items():
        row = scored[key]
        if set(row) != {'context', 'target', 'token_logprobs'} or any(row[k] != req[k] for k in req):
            raise ValueError('score context/target mismatch')
        values[key] = sum_token_logprobs(row['token_logprobs'])
    ids = tuple(sorted(case['candidates']))
    correction = case['background_correction']
    g = tuple(tuple(values[f'{kind}/{cid}'] - (values[f'background/{cid}'] if correction else 0.)
                    for cid in ids) for kind in ('g_positive', 'g_negative'))
    d = tuple(tuple(values[f'd/{cid}/{v}'] - (values[f'd_background/{v}'] if correction else 0.)
                    for v in ('correct', 'incorrect')) for cid in ids)
    return ids, g, d


class HuggingFaceSequenceScorer:
    """Local-only, pinned checkpoint; no implicit download or remote code.

    Boundary convention is explicit: separately tokenize context and target,
    concatenate IDs, prepend BOS if available, score target tokens only.
    No chat template, EOS, truncation, length averaging, or fallback.
    """
    def __init__(self, model_path, revision):
        if not revision:
            raise ValueError('pinned checkpoint content digest required')
        from pathlib import Path
        if not Path(model_path).is_dir():
            raise ValueError('existing local checkpoint directory required')
        if checkpoint_digest(model_path) != revision:
            raise ValueError('checkpoint content digest mismatch')
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        self.model.eval()

    def __call__(self, request):
        if set(request) != {'context', 'target'}:
            raise ValueError('only context and target allowed')
        tok = self.tokenizer
        prefix = tok.encode(request['context'], add_special_tokens=False)
        target = tok.encode(request['target'], add_special_tokens=False)
        if tok.bos_token_id is not None:
            prefix = [tok.bos_token_id] + prefix
        if not prefix or not target:
            raise ValueError('nonempty prefix and target token sequences required')
        ids = prefix + target
        limit = getattr(self.model.config, 'max_position_embeddings', None)
        if limit and len(ids) > limit:
            raise ValueError('context overflow; silent truncation forbidden')
        torch = self.torch
        with torch.inference_mode():
            tensor = torch.tensor([ids], device=self.model.device)
            logits = self.model(input_ids=tensor).logits[0, len(prefix)-1:len(ids)-1, :].float()
            labels = torch.tensor(target, device=logits.device)
            logps = logits.log_softmax(-1).gather(1, labels[:, None]).squeeze(1).tolist()
        sum_token_logprobs(logps)
        return {**request, 'token_logprobs': logps}
