"""Offline sample selection; private labels never imported by the collector."""
import hashlib
from pathlib import Path
from experiments.mechanism.build_upgrade_history import build, normalize
from experiments.mechanism.run_upgrade_pilot import freeze
from experiments.mechanism import redesign_hotpot_data as data

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'experiments/results/action_comparison_v1'


def select(rows, used, n=24):
    # Transitive support-page groups, including unselected bridge questions.
    parent = {r['id']: r['id'] for r in rows}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    pages = {}
    for row in rows:
        for title in row['supporting_facts']['title']:
            page = normalize(title)
            if page in pages:
                parent[find(row['id'])] = find(pages[page])
            pages[page] = row['id']
    blocked = {find(r['id']) for r in rows if normalize(r['question']) in used}
    ranked = sorted(rows, key=lambda r: hashlib.sha256(('action-comparison-v1/' + r['id']).encode()).hexdigest())
    chosen, groups = [], set()
    for row in ranked:
        group = find(row['id'])
        if group in blocked or group in groups:
            continue
        chosen.append(row)
        groups.add(group)
        if len(chosen) == n:
            break
    if len(chosen) != n:
        raise ValueError('insufficient isolated source groups; do not relax rules')
    return chosen, {r['id']: find(r['id']) for r in chosen}


def main():
    if (OUT / 'public_cases.json').exists():
        raise RuntimeError('selection already sealed; do not reselect against expanded history')
    history = build(ROOT)
    if history['errors']:
        raise ValueError('unreadable historical files')
    used = {v['text'] for v in history['texts'].values()}
    rows = data.parquet_module().read_table(data.RAW / 'dev_fullwiki.parquet').to_pylist()
    selected, groups = select(rows, used)
    public, private, splits = [], {}, {}
    for i, row in enumerate(selected):
        docs = []
        for title, sentences in zip(row['context']['title'], row['context']['sentences']):
            text = ''.join(sentences)
            docs.append({'docid': 'native:' + hashlib.sha256(title.encode()).hexdigest(),
                         'title': title, 'text': text, 'text_sha256': hashlib.sha256(text.encode()).hexdigest()})
        # Public material contains no supporting-fact labels or dataset type.
        public.append({'id': row['id'], 'question': row['question'], 'native_documents': docs})
        private[row['id']] = {'answer': row['answer'], 'supporting_facts': row['supporting_facts']}
        splits[row['id']] = 'development' if i < 12 else 'confirmation'
    dev_pages = {normalize(d['title']) for c in public[:12] for d in c['native_documents']}
    test_pages = {normalize(d['title']) for c in public[12:] for d in c['native_documents']}
    OUT.mkdir(parents=True, exist_ok=True)
    freeze(OUT / 'public_cases.json', public)
    freeze(OUT / 'private_labels.json', private)
    freeze(OUT / 'splits.json', splits)
    freeze(OUT / 'selection.json', {'n': 24, 'splits': {'development': 12, 'confirmation': 12},
        'support_components': groups, 'prior_normalized_question_overlap': 0,
        'transitive_support_component_overlap': 0, 'native_page_overlap_count': len(dev_pages & test_pages),
        'history_sha256': {r['path']: r['sha256'] for r in history['files']},
        'raw_dataset_sha256': hashlib.sha256((data.RAW / 'dev_fullwiki.parquet').read_bytes()).hexdigest(),
        'limits': ['history scanner exact normalization, not semantic or pretraining isolation',
                   'support groups refer to matched historical HotpotQA questions, not every previously retrieved page',
                   'native distractors may overlap; no claim that all text sources are disjoint'],
        'labels_sha256': hashlib.sha256((OUT / 'private_labels.json').read_bytes()).hexdigest()})
    print('Prepared 24 questions, 24 distinct transitive support groups; no API calls.')


if __name__ == '__main__':
    main()
