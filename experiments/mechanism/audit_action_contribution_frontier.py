"""Zero-call paired action audit; no pooled significance or policy fitting."""
from hashlib import sha256
import json
from math import comb
from pathlib import Path
from experiments.mechanism.run_upgrade_pilot import freeze

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'experiments/results/resource_allocation'
OUT = ROOT / 'experiments/results/audit/action_contribution_frontier_2026-10-25'


def paired(outputs, labels, fast, thinking):
    rows = []
    for cid, slots in outputs.items():
        a, b = slots[fast], slots[thinking]
        rows.append({'id': cid, 'fast_correct': a['answer'] == labels[cid],
            'thinking_correct': b['answer'] == labels[cid],
            'fast_valid': a['valid'], 'thinking_valid': b['valid'],
            'fast_tokens': a['tokens'], 'thinking_tokens': b['tokens']})
    corrected = [r for r in rows if not r['fast_correct'] and r['thinking_correct']]
    harmed = [r for r in rows if r['fast_correct'] and not r['thinking_correct']]
    n = len(corrected) + len(harmed)
    p = min(1., 2 * sum(comb(n, k) for k in range(min(len(corrected), len(harmed)) + 1)) / 2 ** n) if n else 1.
    return {'n': len(rows), 'fast_correct': sum(r['fast_correct'] for r in rows),
        'thinking_correct': sum(r['thinking_correct'] for r in rows),
        'corrected': len(corrected), 'harmed': len(harmed), 'descriptive_exact_p': p,
        'both_valid_corrected': sum(r['fast_valid'] and r['thinking_valid'] for r in corrected),
        'correction_with_invalid_fast': sum(not r['fast_valid'] for r in corrected),
        'oracle_union_correct_not_deployable': sum(r['fast_correct'] or r['thinking_correct'] for r in rows),
        'fast_tokens': sum(r['fast_tokens'] for r in rows),
        'thinking_tokens': sum(r['thinking_tokens'] for r in rows), 'records': rows}


def build():
    paths = set()
    def load(directory, name):
        p = DATA / directory / name
        paths.add(p)
        return json.loads(p.read_text())
    ca = load('explicit_thinking_control_v1', 'public_cases.json')
    oa = load('explicit_thinking_control_v1', 'sealed_outputs.json')
    la = {v['id']: v['label'] for v in load('two_backend_complementarity_v1', 'private_labels.json')}
    pairs = {}
    for split in ('calibration', 'evaluation', 'all'):
        ids = {c['id'] for c in ca if split == 'all' or c['split'] == split}
        pairs['explicit_' + split] = paired({k: v for k, v in oa.items() if k in ids}, la, 'disabled', 'enabled')
    cb = load('reliable_readout_v1', 'public_cases.json')
    ob = load('reliable_readout_v1', 'sealed_outputs.json')
    lb = {r['id']: r['gold'] for r in load('reliable_readout_v1', 'result.json')['records']}
    pairs['reliable_evaluation'] = paired(ob, lb, 'D0', 'T0')
    cp = []
    for split in ('train', 'test'):
        outputs = load('learned_production_v1', split + '_sealed_outputs.json')
        labels = load('learned_production_v1', split + '_labels.json')
        cp += load('learned_production_v1', split + '_public.json')
        for role in (1, 2):
            pairs[f'production_{split}_role{role}'] = paired(outputs, labels, 'fast' + str(role), 'think' + str(role))
    cohorts = {'explicit36': ca, 'reliable18': cb, 'production24': cp}
    overlaps = {}
    for a, cases_a in cohorts.items():
        for b, cases_b in cohorts.items():
            if a >= b: continue
            overlaps[a + '/' + b] = {'ids': sorted({c['id'] for c in cases_a} & {c['id'] for c in cases_b}),
                'question_hashes': sorted({sha256(c['question'].strip().encode()).hexdigest() for c in cases_a} &
                                         {sha256(c['question'].strip().encode()).hexdigest() for c in cases_b})}
    controllers = load('learned_production_v1', 'result.json')['metrics']
    readouts = load('information_readout_gap_v1', 'result.json')
    return {'new_api_calls': 0, 'scope': 'posthoc historical audit, not fresh validation',
        'paired_strata': pairs, 'cohort_overlap': overlaps,
        'production_test_controllers': controllers, 'readout_metrics': readouts['metrics'],
        'readout_oracle_coverage_not_deployable': readouts['oracle_available_correct'],
        'decision': {'action_positive_evidence': True, 'pure_game_or_aeic_gain_established': False,
                     'coordination_necessity_established': False, 'repeat_same_small_screen': False},
        'limits': ['no pooled p-value; calibration/all and repeated roles are not independent studies',
                   'same backend does not imply equal compute; reliable-readout prompts/caps also differ',
                   'valid means syntax only; oracle union is not a deployable selector',
                   'not a new method or a general impossibility result'],
        'source_sha256': {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = build()
    freeze(OUT / 'result.json', result)
    print(json.dumps(result['cohort_overlap'], ensure_ascii=False))
    for name, row in result['paired_strata'].items():
        print(name, json.dumps({k: v for k, v in row.items() if k != 'records'}, ensure_ascii=False))


if __name__ == '__main__': main()
