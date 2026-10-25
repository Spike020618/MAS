"""Post-hoc original-stem grouping and source-label audit, without rewriting data.

MC proposed answers are separate binary records, but the same original stem is
not an independent source question. Native MC and these binary tasks differ.
"""
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import random
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'experiments/results/audit/legacy_gain_attribution_2026-10-22'


def normalize(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text).casefold()).strip()


def split_question(text):
    if '\nProposed answer:' not in text:
        return text, None
    stem, suffix = text.split('\nProposed answer:', 1)
    # Remove only the known final instruction, not internal answer newlines.
    for marker in ['\nFirst solve the question independently, then judge:',
                   '\nIs this proposed answer factually correct?']:
        suffix = suffix.split(marker, 1)[0]
    return stem.removeprefix('Question: '), suffix.strip()


def answer_key(text):
    # Authors MC snapshot drops final periods present in HF version. This is a
    # second, explicitly punctuation-relaxed check, not semantic fuzzy matching.
    return normalize(text).rstrip('.')


def difference(pairs):
    return {'n': len(pairs), 'left_correct': sum(a for a, b in pairs),
            'right_correct': sum(b for a, b in pairs),
            'wrong_correct': sum(not a and b for a, b in pairs),
            'correct_wrong': sum(a and not b for a, b in pairs)}


def grouped_sensitivity(cache, split, paired):
    by_id = {str(q['question_id']): q for q in cache}
    groups = defaultdict(list)
    for part, ids in split.items():
        for qid in ids:
            stem, _ = split_question(by_id[qid]['question'])
            groups[normalize(stem)].append({'split': part, 'id': qid})
    crossing = {s for s, xs in groups.items() if any(x['split'] == 'test' for x in xs)
                and any(x['split'] != 'test' for x in xs)}
    selected, clean, grouped_pairs, exclusions = [], [], defaultdict(list), []
    seen = set()
    for qid in split['test']:
        stem, _ = split_question(by_id[qid]['question'])
        s = normalize(stem)
        pair = paired[qid]
        grouped_pairs[s].append(pair)
        selected.append(pair)
        # First occurrence by frozen input order, never by outcome or gold.
        if s in crossing or s in seen:
            exclusions.append({'id': qid, 'stem': stem,
                'reason': 'original_stem_crosses_train_dev' if s in crossing else 'later_test_stem_variant'})
        else:
            clean.append(pair)
        seen.add(s)
    # Post-hoc source-cluster bootstrap, equal record weights. Does not remove
    # reputation dependence; intervals exploratory and not confirmatory.
    rng = random.Random(20261022)
    clusters = list(grouped_pairs.values())
    draws = []
    for _ in range(2000):
        sampled = [p for _ in clusters for p in clusters[rng.randrange(len(clusters))]]
        draws.append(sum(int(b)-int(a) for a, b in sampled)/len(sampled))
    draws.sort()
    return {'original': difference(selected),
            'unique_test_stems': len(grouped_pairs),
            'test_records_crossing_train_dev_stem': sum(x['split'] == 'test' for s in crossing for x in groups[s]),
            'grouped_input_clean_first_test_only': difference(clean),
            'excluded_records': exclusions,
            'source_cluster_bootstrap_95_descriptive': [draws[49], draws[1949]],
            'limits': 'saved execution; exclusions do not remove previous history effects; bootstrap not sequentially valid inference'}


def source_labels(dataset, cache, native):
    lookup = defaultdict(list)
    for r in native:
        lookup[normalize(r['question'])].append(r)
    counts, rows = Counter(), []
    for q in cache:
        stem, proposed = split_question(q['question'])
        sources = lookup[normalize(stem)]
        labels, exact = set(), set()
        for r in sources:
            if dataset == 'StrategyQA':
                labels.add('yes' if r['answer'] else 'no')
                exact.update(labels)
            else:
                options = (list(zip(r['choices'], ['yes' if i == r['answer'] else 'no' for i in range(len(r['choices']))]))
                           if dataset == 'MMLU-Hard' else
                           [(text, 'yes' if value else 'no') for text, value in r['mc1_targets'].items()])
                for text, label in options:
                    if normalize(text) == normalize(proposed):
                        exact.add(label)
                    if answer_key(text) == answer_key(proposed):
                        labels.add(label)
        status = ('unmatched_stem' if not sources else 'unmatched_proposal' if not labels else
                  'ambiguous_source_label' if len(labels) > 1 else
                  'label_matches' if q['ground_truth'] in labels else 'label_mismatch')
        counts[status] += 1
        counts['exact_answer_match'] += bool(exact)
        rows.append({'id': str(q['question_id']), 'status': status,
                     'stored_label': q['ground_truth'], 'native_labels': sorted(labels),
                     'exact_answer_match': bool(exact), 'question': q['question']})
    return {'counts': dict(counts), 'rows': rows,
            'scope': 'native dataset labels, not independent factual truth adjudication; final-period relaxation explicit'}


def main():
    import pyarrow.parquet as pq
    split_path = ROOT / 'experiments/mechanism/splits.json'
    splits = json.loads(split_path.read_text())['datasets']
    path = ROOT / 'experiments/results/mechanism/frozen/formal_v10/question_level.jsonl'
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    pairs = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        c = r['condition'].split('__')[1].split('@')[0]
        pairs[r['dataset']][str(r['question_id'])][c] = r['final_decision'] == r['ground_truth']
    source_paths = {
        'StrategyQA': ROOT / 'experiments/results/mechanism/stackelberg/locked_source_game_test_v1/official_train.json',
        'MMLU-Hard': ROOT / 'experiments/data/raw/upgrade_p2_v1/mmlu_test.parquet',
        'TruthfulQA': ROOT / 'experiments/data/raw/legacy_attribution_2026-10-22/truthfulqa_hf_validation.parquet'}
    result = {'scope': 'post-hoc grouping sensitivity and native-label cross-check, zero APIs', 'datasets': {}}
    inputs = [path, split_path, Path(__file__).resolve()]
    for d, sp in splits.items():
        cache_path = ROOT / sp['source']
        cache = json.loads(cache_path.read_text())['questions']
        paired = {qid: (v['no_mechanism'], v['static_aeic']) for qid, v in pairs[d].items()}
        native = (pq.read_table(source_paths[d]).to_pylist() if source_paths[d].suffix == '.parquet'
                  else json.loads(source_paths[d].read_text()))
        if d == 'TruthfulQA':
            if sha256(source_paths[d].read_bytes()).hexdigest() != '23f08e230ca4ed66babf3a72419af7cbde1f3d734dd396ac4cf6d088bd162afd' or len(native) != 817:
                raise ValueError('complete pinned official HF TruthfulQA snapshot required')
            native = [{'question': r['question'], 'mc1_targets': dict(zip(r['mc1_targets']['choices'], r['mc1_targets']['labels']))} for r in native]
        if d == 'StrategyQA':
            dev = ROOT / 'experiments/data/raw/legacy_attribution_2026-10-22/strategyqa_author_dev.json'
            extra = json.loads(dev.read_text())
            if len(extra) != 229 or sha256(dev.read_bytes()).hexdigest() != '0d94842ffb022db8fd5ecd2168b785d5cf67f6faf64f665476bad544a3eb9dde':
                raise ValueError('incomplete author development source')
            native += extra
            inputs.append(dev)
        labels = source_labels(d, cache, native)
        result['datasets'][d] = {'grouping': grouped_sensitivity(cache, sp['splits'], paired),
                                 'native_label_check': labels}
        inputs += [cache_path, source_paths[d]]
    result['inputs_sha256'] = {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in inputs}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'source_groups.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({d: {'grouping': {k:v for k,v in r['grouping'].items() if k != 'excluded_records'},
                          'labels': r['native_label_check']['counts']} for d,r in result['datasets'].items()}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
