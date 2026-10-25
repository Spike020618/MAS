"""Read-only evidence recomputation; writes only a new inventory audit artifact.

Does not call models, alter frozen records, or replace historical audit reports.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def audit():
    frozen = ROOT / 'experiments/results/mechanism/frozen'
    manifest = json.loads((frozen / 'manifest.json').read_text())
    suites = []
    for name in manifest['required_suites']:
        path = frozen / name / 'question_level.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        expected = next(b for b in manifest['bundles'] if b['name'] == name)
        suites.append({'suite': name, 'records': len(rows),
            'sha256_matches_manifest': hashlib.sha256(path.read_bytes()).hexdigest() == expected['question_level_sha256'],
            'expected_count_matches': len(rows) == expected['n_question_level_records']})
    path = frozen / 'formal_v10/question_level.jsonl'
    main = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    grouped = defaultdict(list)
    for row in main:
        grouped[row['condition']].append(row)
    conditions = {}
    paired = {}
    for name, rows in grouped.items():
        actions, choices, executions = Counter(), Counter(), Counter()
        correct = 0
        for row in rows:
            key = (row['dataset'], row['question_id'])
            if key in paired.setdefault(name, {}):
                raise ValueError('duplicate question-condition')
            paired[name][key] = row['final_decision'] == row['ground_truth']
            correct += paired[name][key]
            for turn in row['rounds']:
                actions[turn['chosen_action']] += 1
                for trace in turn['action_trace']:
                    target = choices if trace['stage'] == 'choose' else executions
                    target[f"{trace.get('backend')}/{trace.get('model')}/{trace['stage']}"] += 1
        conditions[name] = {'question_records': len(rows), 'correct': correct,
            'accuracy': correct / len(rows), 'actions': dict(actions),
            'choice_call_routes': dict(choices), 'execution_call_routes': dict(executions)}
    base = paired['independent__no_mechanism@0.2']
    treatment = paired['independent__static_aeic@0.2']
    same = base.keys() == treatment.keys()
    if not same:
        raise ValueError('main paired question sets differ')
    flow = Counter(('correct' if base[k] else 'wrong') + '_to_' +
                   ('correct' if treatment[k] else 'wrong') for k in base)
    interface = sum(bool(l.strip()) for l in (ROOT / 'experiments/results/mechanism/interface/report_interface_v1.jsonl').read_text().splitlines())
    decomposition = sum(sum(bool(l.strip()) for l in p.read_text().splitlines())
        for p in (ROOT / 'experiments/results/mechanism/decomposition').glob('*.jsonl'))
    checks = {'all_suite_hashes_match': all(s['sha256_matches_manifest'] for s in suites),
        'all_suite_counts_match': all(s['expected_count_matches'] for s in suites),
        'frozen_5850': sum(s['records'] for s in suites) == 5850,
        'main_accuracy_counts': conditions['independent__no_mechanism@0.2']['correct'] == 306
            and conditions['independent__static_aeic@0.2']['correct'] == 370,
        'main_pairs_match': same,
        'paired_error_flow': flow['wrong_to_correct'] == 78 and flow['correct_to_wrong'] == 14,
        'interface_270': interface == 270, 'decomposition_240': decomposition == 240}
    return {'scope': 'offline inventory and recorded-outcome recomputation; not answer-key truth validation or causal attribution',
        'checks': checks, 'passed': all(checks.values()), 'suites': suites,
        'record_counts': {'frozen': sum(s['records'] for s in suites), 'interface': interface,
                          'decomposition': decomposition, 'total': sum(s['records'] for s in suites)+interface+decomposition},
        'main_conditions': conditions, 'paired_error_flow': dict(flow),
        'important_boundary': 'DeepSeek choice backend does not imply DeepSeek-only execution; resource-tiered routes already include GLM and reasoner.'}


if __name__ == '__main__':
    result = audit()
    out = ROOT / 'experiments/results/mechanism/synthesis/work_inventory_audit_2026-10-21.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result['passed']:
        raise SystemExit(1)
