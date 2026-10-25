"""Conservative historical text/ID exclusion index. Offline; never modifies inputs.

This is an exact-normalization screen, NOT a semantic rewrite detector. Unknown
schemas and unreadable files remain explicit blockers to an unseen-data claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import unicodedata


def normalize(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text).casefold()).strip()


def walk(value, path='$'):
    if isinstance(value, dict):
        fields = ('question', 'question_text', 'base_question', 'original_question', 'prompt_question')
        for field in fields:
            text = value.get(field)
            if isinstance(text, str) and text.strip():
                yield {'text': normalize(text), 'location': path + '.' + field,
                       'dataset': str(value.get('dataset', value.get('source', ''))),
                       'id': str(value.get('original_id', value.get('question_id', value.get('id', ''))))}
        for key, child in value.items():
            if isinstance(child, (dict, list)):
                yield from walk(child, path + '.' + str(key))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from walk(child, path + '[' + str(i) + ']')


def build(root):
    paths = set((root / 'experiments/results').rglob('*.json'))
    paths.update((root / 'experiments/results').rglob('*.jsonl'))
    paths.update((root / 'experiments').glob('*cache*.json'))
    paths.update((root / 'experiments/data/processed').rglob('*.json'))
    texts, ids, files, errors = {}, {}, [], []
    seen_hashes = {}
    pool_path = root / 'experiments/results/mechanism/stackelberg/locked_source_game_test_v1/official_train.json'
    pool_provenance = pool_path.parent / 'dataset_provenance.json'
    pool_hash = None
    pool_canonical_hash = None
    if pool_path.exists() and pool_provenance.exists():
        provenance = json.loads(pool_provenance.read_text())
        if (provenance.get('url') == 'https://raw.githubusercontent.com/eladsegal/strategyqa/main/data/strategyqa/train.json'
                and provenance.get('sha256') == hashlib.sha256(pool_path.read_bytes()).hexdigest()
                and provenance.get('n_examples') == len(json.loads(pool_path.read_text()))):
            pool_hash = provenance['sha256']
            pool_canonical_hash = hashlib.sha256(json.dumps(json.loads(pool_path.read_text()),
                                                           sort_keys=True).encode()).hexdigest()
    for path in sorted(paths):
        relative = str(path.relative_to(root))
        if '/upgrade_game/p0/' in relative or '/audit/' in relative:
            continue
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        entry = {'path': relative, 'sha256': sha, 'bytes': len(raw)}
        is_pool_copy = False
        if pool_canonical_hash and path.name == 'source_snapshot.json':
            try:
                candidate = json.loads(raw)
                is_pool_copy = hashlib.sha256(json.dumps(candidate, sort_keys=True).encode()).hexdigest() == pool_canonical_hash
            except ValueError:
                pass
        if sha == pool_hash or is_pool_copy:
            entry['role'] = 'verified_official_raw_pool_not_model_execution'
            entry['question_fields'] = 0
            entry['coverage'] = 'retained provenance; original pool is not evidence of per-question API use'
            files.append(entry)
            continue
        if sha in seen_hashes:
            entry['duplicate_of'] = seen_hashes[sha]
            files.append(entry)
            continue
        seen_hashes[sha] = relative
        try:
            values = [json.loads(line) for line in raw.decode().splitlines() if line.strip()] if path.suffix == '.jsonl' else [json.loads(raw)]
            count = 0
            for value in values:
                for row in walk(value):
                    text_hash = hashlib.sha256(row['text'].encode()).hexdigest()
                    evidence = {'path': relative, 'location': row['location'], 'dataset': row['dataset'], 'id': row['id']}
                    texts.setdefault(text_hash, {'text': row['text'], 'examples': []})
                    if len(texts[text_hash]['examples']) < 3:
                        texts[text_hash]['examples'].append(evidence)
                    if row['dataset'] and row['id']:
                        ids.setdefault(row['dataset'] + ':' + row['id'], text_hash)
                    count += 1
            entry['question_fields'] = count
            if not count:
                entry['coverage'] = 'no recognized question field; manual/schema review required'
        except (ValueError, UnicodeError) as exc:
            errors.append({'path': relative, 'error': type(exc).__name__})
        files.append(entry)
    return {'version': 'upgrade-history-v2', 'files': files, 'texts': texts, 'ids': ids,
            'errors': errors, 'unique_texts': len(texts),
            'limitation': 'Exact NFKC/case/whitespace screening only; no semantic paraphrase guarantee. '
                          'Files without recognized question fields require schema review before freshness certification.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'files': len(result['files']), 'unique_texts': result['unique_texts'],
                      'ids': len(result['ids']), 'errors': result['errors']}, ensure_ascii=False))
