"""Label-isolated 128-question retrieval coverage diagnostic, no model calls."""
import argparse
from hashlib import sha256
from pathlib import Path
import time
import unicodedata
from experiments.core.mechanism.evidence_acquisition import Corpus
from experiments.core.mechanism.title_anchor_retrieval import TitleAnchorCorpus
from experiments.mechanism import run_open_evidence_screen_v2 as p
from experiments.mechanism.build_upgrade_history import build, normalize
from experiments.mechanism.run_upgrade_pilot import freeze, lock

ROOT = p.ROOT
OUT = ROOT / 'experiments/results/redesign_v1/retrieval_coverage_v1'
DESIGN = ROOT / 'docs/research/redesign_v1/检索覆盖诊断_预定协议_2026-10-25.md'


def title_key(text): return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'protocol.json').exists(): verify(); return
    history = build(ROOT)
    if history['errors']: raise ValueError('history unreadable')
    used = {v['text'] for v in history['texts'].values()}
    source = p.data.parquet_module().read_table(p.data.RAW / 'dev_fullwiki.parquet', columns=['id', 'question']).to_pylist()
    source.sort(key=lambda r: sha256(('coverage-v1/' + r['id']).encode()).hexdigest())
    chosen = [r for r in source if normalize(r['question']) not in used][:128]
    if len(chosen) != 128: raise ValueError('insufficient unused inputs')
    cases = [{**r, 'split': 'diagnostic' if i < 64 else 'confirmation'} for i, r in enumerate(chosen)]
    freeze(OUT / 'public_cases.json', cases)
    freeze(OUT / 'selection.json', {'history_files': len(history['files']),
        'history_sha256': {v['path']: v['sha256'] for v in history['files']},
        'exclusion': 'exact normalized question; not pretraining or semantic novelty'})
    files = [Path(__file__), ROOT / 'experiments/core/mechanism/title_anchor_retrieval.py',
        ROOT / 'experiments/core/mechanism/evidence_acquisition.py', DESIGN, OUT / 'public_cases.json']
    freeze(OUT / 'protocol.json', {'n': 128, 'model_calls': 0, 'top_k': [2, 4, 8, 10],
        'sha256': {str(f.relative_to(ROOT)): p.base.h(f) for f in files},
        'corpus_manifest_sha256': p.base.h(p.data.ENV / 'index_manifest.json')})


def verify():
    protocol = p.base.read(OUT / 'protocol.json')
    for path, value in protocol['sha256'].items():
        if p.base.h(ROOT / path) != value: raise ValueError('frozen retrieval diagnostic changed:' + path)
    if protocol['corpus_manifest_sha256'] != p.base.h(p.data.ENV / 'index_manifest.json'): raise ValueError('corpus changed')


def retrieve():
    verify()
    with lock(OUT):
        cases = p.base.read(OUT / 'public_cases.json'); outputs = {}
        baseline = Corpus(p.data.ENV / 'wiki_fts.sqlite'); anchored = TitleAnchorCorpus(p.data.ENV / 'wiki_fts.sqlite')
        try:
            for i, case in enumerate(cases):
                path = OUT / (case['id'] + '.json')
                if path.exists(): outputs[case['id']] = p.base.read(path); continue
                result = {}
                for name, engine in (('bm25', baseline), ('title_anchor', anchored)):
                    started = time.perf_counter(); docs = engine.search(case['question'], 10)
                    result[name] = {'documents': [{k: d[k] for k in ('docid', 'title', 'text_sha256')} for d in docs],
                        'wall_seconds': time.perf_counter() - started}
                freeze(path, result); outputs[case['id']] = result
                if (i + 1) % 16 == 0: print('retrieval sealed', i + 1, '/128', flush=True)
        finally: baseline.close(); anchored.close()
        freeze(OUT / 'sealed_retrieval.json', outputs)


def analyze():
    verify(); cases = p.base.read(OUT / 'public_cases.json'); outputs = p.base.read(OUT / 'sealed_retrieval.json')
    if set(outputs) != {c['id'] for c in cases}: raise ValueError('incomplete retrieval; do not open support labels')
    source = p.data.parquet_module().read_table(p.data.RAW / 'dev_fullwiki.parquet', columns=['id', 'supporting_facts']).to_pylist()
    supports = {r['id']: sorted({title_key(t) for t in r['supporting_facts']['title']}) for r in source if r['id'] in outputs}
    wanted = {t for ts in supports.values() for t in ts}; found = set()
    for batch in p.data.parquet_module().ParquetFile(p.data.RAW / 'corpus.parquet').iter_batches(batch_size=20000, columns=['title']):
        for title in batch.column(0).to_pylist():
            key = title_key(title)
            if key in wanted: found.add(key)
    freeze(OUT / 'offline_supports.json', {'supports': supports, 'corpus_present': sorted(found), 'corpus_missing': sorted(wanted - found)})
    metrics = {}
    for split in ('diagnostic', 'confirmation', 'all'):
        selected = [c['id'] for c in cases if split == 'all' or c['split'] == split]; n = len(selected)
        stats = {'n': n, 'all_support_titles_available': sum(set(supports[c]) <= found for c in selected)}
        for name in ('bm25', 'title_anchor'):
            cell = {'sum_search_seconds': sum(outputs[c][name]['wall_seconds'] for c in selected)}
            for k in (2, 4, 8, 10):
                overlaps = [len({title_key(d['title']) for d in outputs[c][name]['documents'][:k]} & set(supports[c])) for c in selected]
                cell[str(k)] = {'mean_support_title_recall': sum(v / len(supports[c]) for c, v in zip(selected, overlaps)) / n,
                    'all_support_titles_covered': sum(v == len(supports[c]) for c, v in zip(selected, overlaps)),
                    'at_least_one_support_title': sum(v > 0 for v in overlaps)}
            stats[name] = cell
        metrics[split] = stats
    freeze(OUT / 'result.json', {'scope': 'retrieval title coverage, NOT answer accuracy or game benefit', 'metrics': metrics, 'model_calls': 0})
    print(__import__('json').dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    a = argparse.ArgumentParser(); a.add_argument('stage', choices=('prepare', 'retrieve', 'analyze'))
    {'prepare': prepare, 'retrieve': retrieve, 'analyze': analyze}[a.parse_args().stage]()
