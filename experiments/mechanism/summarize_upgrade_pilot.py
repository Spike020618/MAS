"""Label-free pilot diagnostics; run only after independent integrity audit."""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import statistics

from experiments.core.mechanism.upgrade_game import aggregate, digest
from experiments.mechanism.audit_upgrade_pilot import audit
from experiments.mechanism.run_upgrade_pilot import freeze, parse, read_rows


def summarize(directory):
    checked = audit(directory)
    cases = json.loads((directory / 'public_cases.json').read_text())
    raw_rows = read_rows(directory / 'responses.jsonl')
    rows = {r['key']: r for r in raw_rows}
    timeline = read_rows(directory / 'timeline.jsonl')
    timestamps = {(r['key'], r['event']): datetime.fromisoformat(r['utc']) for r in timeline}
    conditions, details = Counter(), []
    for case in cases:
        prefix = digest(case)[:20]
        initial = [parse(rows[prefix + '/initial/' + str(i)], case['choices'])[0] for i in range(3)]
        final = json.loads((directory / (prefix + '.sealed.json')).read_text())
        full = [parse(rows[prefix + ('/upgrade/' if a else '/diagnostic/') + str(i)], case['choices'])[0]
                for i, a in enumerate(final['actions'])]
        votes = [r['C'] for r in initial]
        condition = 'unanimous' if len(set(votes)) == 1 else 'disagreement'
        if 'abstain' in votes:
            condition += '_with_abstention'
        conditions[condition] += 1
        initial_decision = aggregate(initial, case['choices'])
        details.append({'case_id': case['id'], 'initial_state': condition,
                        'selected_actions': final['actions'],
                        'initial_decision': initial_decision, 'selected_decision': final['decision'],
                        'full_upgrade_decision': aggregate(full, case['choices']),
                        'selected_decision_changed': initial_decision != final['decision'],
                        'individual_upgrades_changed': sum(a['C'] != b['C'] for a, b in zip(initial, full)),
                        'deployed_end_to_end_wall_seconds': (
                            timestamps[(prefix, 'seal')] - timestamps[(prefix + '/initial/0', 'attempt')]).total_seconds(),
                        'state_to_selection_wall_seconds_including_checkpoint_io': (
                            timestamps[(prefix, 'selection')] - timestamps[(prefix, 'state')]).total_seconds()})
    usage = [json.loads(r['response']['body'])['usage'] for r in raw_rows]
    bodies = [json.loads(r['response']['body']) for r in raw_rows]
    return {'integrity_audit': checked, 'labels_opened': False,
            'scope': 'engineering pilot with synthetic public payoff table; no efficacy conclusion',
            'cases': len(cases), 'responses': len(raw_rows),
            'requested_models': sorted({r['response']['requested_model'] for r in raw_rows}),
            'returned_models': sorted({b['model'] for b in bodies}),
            'provider_fingerprints': sorted({str(b.get('system_fingerprint')) for b in bodies}),
            'usage': {k: sum(u[k] for u in usage) for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')},
            'initial_states': dict(conditions),
            'selected_decision_changes': sum(r['selected_decision_changed'] for r in details),
            'individual_upgrade_changes': sum(r['individual_upgrades_changed'] for r in details),
            'median_deployed_wall_seconds': statistics.median(r['deployed_end_to_end_wall_seconds'] for r in details),
            'details': details}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    result = summarize(args.directory)
    freeze(args.directory / 'label_free_diagnostics.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'details'}))
