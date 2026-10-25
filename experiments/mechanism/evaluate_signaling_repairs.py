"""Two principled offline controls. Not lambda search or independent validation."""
import json
from pathlib import Path
from experiments.core.mechanism.signaling_repair_controls import frozen_checker_controls
from experiments.mechanism.run_upgrade_pilot import freeze

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'experiments/results/resource_allocation/signaling_pilot_v1'
OUT=ROOT/'experiments/results/audit/signaling_failure_repair_v2'


def main():
    cases=json.loads((SOURCE/'public_cases.json').read_text())
    readouts=json.loads((SOURCE/'sealed_readouts.json').read_text())
    results={}
    for c in cases:
        details=readouts[c['id']]['details']
        if details:
            ref=details['game_reference']
            results[c['id']]=frozen_checker_controls(c['choices'],ref['generator'],ref['discriminator'])
        else:results[c['id']]={'answers':{'freeze_checker':'abstain','uniform_anchor_freeze_checker':'abstain'},
                             'scope':'input unavailable, no replacement probabilities'}
    freeze(OUT/'structural_controls_label_blind.json',results)
    labels={r['id']:r['gold'] for r in json.loads((SOURCE/'result.json').read_text())['records']}
    summary={'scope':'posthoc structural controls, no independent effect validation','new_api_calls':0,
        'all12_retained':True,'full_input_cases':sum(bool(readouts[c['id']]['details']) for c in cases),
        'correct':{mode:sum(r['answers'][mode]==labels[qid] for qid,r in results.items()) for mode in
                   ('freeze_checker','uniform_anchor_freeze_checker')},
        'uniform_equals_checker_every_usable_case':all(r.get('uniform_readout_equals_checker',True) for r in results.values()),
        'takeaway':'freezing checker alone did not repair quality; uniform prior makes readout equivalent to direct checker, not novel game gain'}
    freeze(OUT/'structural_controls_result.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
