"""Input-only sampling from a verified official StrategyQA snapshot.

Conservative scan also checks question substrings inside raw historical messages,
not just fields recognized by the historical index. No answer-based selection.
"""
import argparse
import hashlib
import json
from pathlib import Path
import unicodedata

from experiments.core.mechanism.upgrade_game import digest
from experiments.mechanism.run_upgrade_pilot import freeze


def canonical(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text).casefold() if c.isalnum())


def strings(value):
    if isinstance(value, str):
        yield value
        if value.lstrip().startswith(('{', '[')):
            try:
                yield from strings(json.loads(value))
            except ValueError:
                pass
    elif isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from strings(child)


def prepare(raw_path, history_path, output, n=12, source_split='train'):
    if not 1 <= n <= 12:
        raise ValueError('pilot size is bounded at 12')
    root = Path(__file__).resolve().parents[2]
    raw = raw_path.read_bytes()
    source = json.loads(raw)
    if source_split not in ('train', 'dev'):
        raise ValueError('official repository split required')
    if not isinstance(source, list) or len(source) < (1000 if source_split == 'train' else 200):
        raise ValueError('incomplete official source snapshot')
    history = json.loads(history_path.read_text())
    if history['errors']:
        raise ValueError('historical parse errors unresolved')
    source_ids = set()
    for row in source:
        if not isinstance(row.get('qid'), str) or not isinstance(row.get('question'), str) or type(row.get('answer')) is not bool:
            raise ValueError('official StrategyQA schema mismatch')
        if row['qid'] in source_ids:
            raise ValueError('duplicate original source ID')
        source_ids.add(row['qid'])
    indexed = {canonical(v['text']) for v in history['texts'].values()}
    # Rank does not use labels, correctness, initial reports, or requested resource actions.
    ranked = sorted(source, key=lambda r: digest(['MAS-upgrade-engineering-2026-10-22-v1', r['qid']]))
    candidates, exclusions = [], []
    seen = set()
    for row in ranked:
        text = canonical(row['question'])
        if text in indexed or text in seen:
            exclusions.append({'source_id': row['qid'], 'reason': 'indexed_or_duplicate_base_question'})
            continue
        seen.add(text)
        candidates.append(row)
        if len(candidates) >= 4 * n:
            break
    blocked = {}
    coverage = []
    for file in history['files']:
        path = root / file['path']
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != file['sha256']:
            raise ValueError('historical file changed since index: ' + file['path'])
        if file.get('role') == 'verified_official_raw_pool_not_model_execution':
            coverage.append({'path': file['path'], 'sha256': file['sha256'],
                             'role': 'verified_official_raw_pool_not_model_execution'})
            continue
        if file.get('duplicate_of'):
            continue
        values = [json.loads(line) for line in data.decode().splitlines() if line.strip()] if path.suffix == '.jsonl' else [json.loads(data)]
        # All string leaves, including escaped JSON messages and dictionary keys.
        flattened = list(strings(values))
        leaf_ids = set(flattened)
        text = '\n'.join(canonical(s) for s in flattened)
        for row in candidates:
            if row['qid'] in blocked:
                continue
            if row['qid'] in leaf_ids or canonical(row['question']) in text:
                blocked[row['qid']] = file['path']
        coverage.append({'path': file['path'], 'sha256': file['sha256']})
    eligible = [r for r in candidates if r['qid'] not in blocked]
    if len(eligible) < n:
        raise ValueError('insufficient screened new questions; increase candidate pool in a versioned revision')
    chosen = eligible[:n]
    public = [{'id': 'strategyqa:' + r['qid'], 'source_id': r['qid'],
               'question': r['question'], 'choices': ['yes', 'no']} for r in chosen]
    private = [{'id': 'strategyqa:' + r['qid'], 'label': 'yes' if r['answer'] else 'no'} for r in chosen]
    provenance = {'source_url': 'https://github.com/eladsegal/strategyqa/raw/refs/heads/main/data/strategyqa/' + source_split + '.json',
                  'source_sha256': hashlib.sha256(raw).hexdigest(), 'source_count': len(source),
                  'source_split': 'authors repository unofficial ' + source_split + ' split of original training set',
                  'history_index_sha256': hashlib.sha256(history_path.read_bytes()).hexdigest(),
                  'scanned_unique_files': len(coverage), 'history_scan_hash': digest(coverage),
                  'public_cases_hash': digest(public), 'n': n,
                  'candidate_count': len(candidates), 'message_collisions': blocked,
                  'initial_exclusions': exclusions,
                  'sampling': 'source-ID hash ranking before any output; no labels used to select',
                  'freshness_scope': 'no exact normalized source question/ID found in indexed historical JSON/JSONL strings; '
                                     'verified raw source-pool snapshots excluded from execution exposure; '
                                     'not a semantic paraphrase, researcher-visibility, or model-pretraining exclusion guarantee'}
    output.mkdir(parents=True, exist_ok=True)
    freeze(output / 'public_cases.json', public)
    freeze(output / 'private_labels.json', private)
    freeze(output / 'provenance.json', provenance)
    freeze(output / 'source_snapshot.json', source)
    return provenance


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--history', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-split', choices=['train', 'dev'], default='train')
    args = parser.parse_args()
    result = prepare(args.raw, args.history, args.output, source_split=args.source_split)
    print(json.dumps({k: result[k] for k in ('source_count', 'n', 'candidate_count', 'scanned_unique_files')}))
