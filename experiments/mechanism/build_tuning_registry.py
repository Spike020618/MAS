"""Retrospective development ledger; never represents it as preregistration."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'experiments/results/mechanism/stackelberg'
STAGES=[
    ('feasibility_pilot_v1','exploratory_diagnostic','four-action execution and payoff-table comparison'),
    ('verified_contract_pilot_v1','development_failure','correctness payment and paid source access'),
    ('calculator_contract_pilot_v2','development_failure','comparison-before-action and arithmetic assistance'),
    ('calculator_contract_pilot_v3','development_validation','complete execution context and explicit C semantics'),
    ('calculator_contract_final_v4','locked_synthetic_parameter_test','fixed v3 method; new costs and reward multipliers'),
]


def main():
    stages=[]
    for name,role,change in STAGES:
        directory=BASE/name
        files={}
        for filename in ['protocol.json','calls.jsonl','summary.json','acceptance.json','posthoc_arithmetic.jsonl']:
            path=directory/filename
            if path.exists():
                files[filename]={'path':str(path.relative_to(ROOT)),
                    'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        summary_path=directory/'summary.json'
        if not summary_path.exists():
            stages.append({'stage':name,'role':role,'status':'not_complete','files':files})
            continue
        summary=json.loads(summary_path.read_text())
        metrics={k:v for k,v in summary.items() if k in [
            'n_decisions','n_trials','api_calls','autonomous_best_reply_rate','explicit_payoff_best_reply_rate',
            'best_reply_rate','paired_switch_rate','gate_pass','summaries']}
        stages.append({'stage':name,'role':role,'change':change,'status':'complete',
                       'metrics':metrics,'files':files,
                       'allowed_claim':'synthetic/interface diagnostic only; no real-QA equilibrium' if name!='feasibility_pilot_v1'
                       else 'small real-QA diagnostic; not equilibrium or expected-payoff estimation'})
    report={'registry_type':'retrospective tuning history, not initial preregistration',
            'selection_history':'v3 selected after observing failures in earlier versions',
            'historical_source_provenance':'exact prompts retained in calls; historical source snapshots not available for every stage',
            'final_test_policy':'preserve final result regardless of success; further changes require a new development cycle and new test',
            'primary_original_qa_results_changed':False,
            'contract_search_note':'original finite grid selection is observed-policy optimization, not equilibrium evidence',
            'stages':stages}
    path=BASE/'tuning_registry_2026-10-18.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(path)


if __name__=='__main__': main()
