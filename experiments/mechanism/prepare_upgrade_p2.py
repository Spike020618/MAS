"""Input-only, grouped calibration/development preparation; no API calls."""
import argparse
import hashlib
import json
from pathlib import Path

from experiments.core.mechanism.upgrade_game import digest
from experiments.mechanism.prepare_upgrade_pilot import canonical, strings
from experiments.mechanism.run_upgrade_pilot import freeze


def native_mc(domain, source_id, question, options, correct):
    if correct not in range(len(options)) or len(set(options)) != len(options):
        raise ValueError('native MC schema invalid')
    # Label-blind order avoids TruthfulQA MC1's correct-answer-first file layout.
    ordered = sorted(range(len(options)), key=lambda j: digest([question, options[j]]))
    names = ['option_' + str(i) for i in range(len(options))]
    rendered = question + '\nOptions:\n' + '\n'.join(names[i] + ': ' + options[j] for i, j in enumerate(ordered))
    return {'domain': domain, 'source_id': source_id, 'base_question': question,
            'question': rendered, 'choices': names, 'label': names[ordered.index(correct)]}


def pool(args):
    rows, sources = [], []
    if args.strategyqa:
        data = json.loads(args.strategyqa.read_text())
        if len(data) < 1000:
            raise ValueError('incomplete StrategyQA source')
        rows += [{'domain': 'strategyqa', 'source_id': r['qid'], 'base_question': r['question'],
                  'question': r['question'], 'choices': ['yes', 'no'],
                  'label': 'yes' if r['answer'] else 'no'} for r in data]
        sources.append({'domain': 'strategyqa', 'path': str(args.strategyqa), 'sha256': hashlib.sha256(args.strategyqa.read_bytes()).hexdigest(), 'n': len(data)})
    if args.truthfulqa:
        data = json.loads(args.truthfulqa.read_text())
        if len(data) != 790 or hashlib.sha256(args.truthfulqa.read_bytes()).hexdigest() != 'c90d853454fd48c676accb33e16923ac438dbb6fb37ecc7c6a70e33c9e556e87':
            raise ValueError('complete pinned authors TruthfulQA MC snapshot required (790 questions)')
        for r in data:
            options = list(r['mc1_targets'])
            correct = [i for i, o in enumerate(options) if r['mc1_targets'][o] == 1]
            if len(correct) != 1:
                raise ValueError('MC1 must have exactly one native correct label')
            rows.append(native_mc('truthfulqa', digest(r['question'])[:24], r['question'], options, correct[0]))
        sources.append({'domain': 'truthfulqa', 'path': str(args.truthfulqa), 'sha256': hashlib.sha256(args.truthfulqa.read_bytes()).hexdigest(), 'n': len(data)})
    if args.mmlu:
        import pandas as pd
        data = pd.read_parquet(args.mmlu).to_dict('records')
        if len(data) < 1000:
            raise ValueError('complete official MMLU test/validation split required')
        excluded = []
        for r in data:
            if len(set(r['choices'])) != len(r['choices']):
                excluded.append({'source_id': digest([r['subject'], r['question']])[:24], 'reason': 'duplicate option text'})
                continue
            rows.append(native_mc('mmlu', digest([r['subject'], r['question']])[:24],
                                  r['question'], list(r['choices']), int(r['answer'])))
        sources.append({'domain': 'mmlu', 'path': str(args.mmlu), 'sha256': hashlib.sha256(args.mmlu.read_bytes()).hexdigest(), 'n': len(data), 'input_schema_exclusions': excluded})
    return rows, sources


def prepare(args):
    root = Path(__file__).resolve().parents[2]
    rows, sources = pool(args)
    if not rows:
        raise ValueError('no verified original pools available')
    history = json.loads(args.history.read_text())
    if history['errors']:
        raise ValueError('history parse errors')
    used_texts = {canonical(v['text']) for v in history['texts'].values()}
    # Explicit P1 exclusion also protects against a stale index supplied accidentally.
    prior = json.loads((root / 'experiments/results/upgrade_game/pilot_data_v1/public_cases.json').read_text())
    used_texts.update(canonical(r['question']) for r in prior)
    prior_ids = {r['source_id'] for r in prior}
    candidates = []
    for domain in sorted({r['domain'] for r in rows}):
        ranked = sorted([r for r in rows if r['domain'] == domain],
                        key=lambda r: digest(['upgrade-P2-2026-10-22-v1', r['source_id']]))
        seen, chosen = set(), []
        for r in ranked:
            key = canonical(r['base_question'])
            if key in seen or key in used_texts or r['source_id'] in prior_ids:
                continue
            seen.add(key); chosen.append(r)
            if len(chosen) == 180:
                break
        candidates.extend(chosen)
    blocked, coverage = {}, []
    for f in history['files']:
        path = root / f['path']
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != f['sha256']:
            raise ValueError('historical snapshot changed: ' + f['path'])
        if f.get('role') == 'verified_official_raw_pool_not_model_execution' or f.get('duplicate_of'):
            continue
        data = [json.loads(s) for s in raw.decode().splitlines() if s.strip()] if path.suffix == '.jsonl' else json.loads(raw)
        leaves = list(strings(data)); leaf_ids = set(leaves)
        content = '\n'.join(canonical(s) for s in leaves)
        for r in candidates:
            key = r['domain'] + ':' + r['source_id']
            if key not in blocked and (r['source_id'] in leaf_ids or canonical(r['base_question']) in content):
                blocked[key] = f['path']
        coverage.append({'path': f['path'], 'sha256': f['sha256']})
    selected = {'calibration': [], 'development': []}
    labels = {'calibration': [], 'development': []}
    groups = []
    used = set()
    for domain in sorted({r['domain'] for r in rows}):
        available = [r for r in candidates if r['domain'] == domain and domain + ':' + r['source_id'] not in blocked]
        if len(available) < 60:
            raise ValueError('insufficient screened cases for ' + domain)
        for i, r in enumerate(available[:60]):
            text = canonical(r['base_question'])
            if text in used:
                raise ValueError('cross-domain original question collision')
            used.add(text)
            split = 'calibration' if i < 40 else 'development'
            case = {'id': domain + ':' + r['source_id'], 'source_id': r['source_id'],
                    'question': r['question'], 'choices': r['choices']}
            selected[split].append(case)
            labels[split].append({'id': case['id'], 'label': r['label']})
            groups.append({'id': case['id'], 'domain': domain, 'split': split, 'base_question': r['base_question']})
    args.output.mkdir(parents=True, exist_ok=True)
    for split in selected:
        # Round-robin domains after deterministic selection avoids a model-time/domain confound.
        selected[split].sort(key=lambda r: digest(['execution-order-v1', r['id']]))
        freeze(args.output / (split + '_public.json'), selected[split])
        freeze(args.output / (split + '_private.json'), labels[split])
    provenance = {'version': 'upgrade-P2-data-v1', 'sources': sources,
                  'domains': sorted({r['domain'] for r in rows}),
                  'scope': 'complete_three_domain_P2' if len(sources) == 3 else 'partial_domain_fragment_NOT_complete_P2',
                  'history_index_sha256': hashlib.sha256(args.history.read_bytes()).hexdigest(),
                  'history_scan_hash': digest(coverage), 'scanned_unique_files': len(coverage),
                  'collisions': blocked, 'groups': groups,
                  'public_hashes': {s: digest(v) for s, v in selected.items()},
                  'private_hashes': {s: digest(v) for s, v in labels.items()},
                  'counts': {s: len(v) for s, v in selected.items()},
                  'freshness_limit': 'source groups and exact normalized question/ID exposure; no paraphrase/pretraining novelty guarantee'}
    freeze(args.output / 'provenance.json', provenance)
    return {k: provenance[k] for k in ('scope', 'domains', 'counts', 'scanned_unique_files')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategyqa', type=Path)
    parser.add_argument('--truthfulqa', type=Path)
    parser.add_argument('--mmlu', type=Path)
    parser.add_argument('--history', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    print(json.dumps(prepare(parser.parse_args())))
