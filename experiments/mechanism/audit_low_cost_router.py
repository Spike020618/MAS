"""Audit routing signal and freeze a candidate, without claiming a new game."""
import hashlib
import json
import re
import subprocess
import sys

from experiments.mechanism import probe_resource_allocation as p
from experiments.mechanism.replay_low_cost_router import OUT,vote,evaluate
from experiments.mechanism.sensitivity_resource_allocation import explicit_answer
from experiments.mechanism.run_upgrade_pilot import read_rows,freeze
from experiments.mechanism.complete_legacy_gain_audit import frozen_hashes


def audit():
    result=evaluate();cases=json.loads((p.OUT/'public_cases.json').read_text())
    rows=read_rows(p.OUT/'responses.jsonl');index={r['key']:r for r in rows}
    labels={c['id']:c['label'] for c in json.loads((p.ROOT/'experiments/results/upgrade_game/p2_data_v2/development_private.json').read_text())}
    groups={};pairs={'prediction_equal':0,'gate_correct_high_wrong':0,'gate_wrong_high_correct':0}
    for case,row in zip(cases,result['policies']['not_unanimous']['rows']):
        if case['id']!=row['id']:raise ValueError('case alignment mismatch')
        low=row['observed_low']
        kind=('invalid_or_abstain' if any(v is None or v=='abstain' for v in low) else
              'valid_disagreement' if len(set(low))>1 else 'valid_unanimous')
        group=groups.setdefault(kind,{'n':0,'high_correct':0,'low_correct_bounds':[0,0]})
        group['n']+=1;group['high_correct']+=row['bounds']['all_high'][0]
        for j in [0,1]:group['low_correct_bounds'][j]+=row['bounds']['all_low'][j]
        high=[explicit_answer(p.body(index[p.digest(case)[:20]+'/deepseek/'+str(a)])['choices'][0]['message']['content'],case['choices']) for a in range(3)]
        hp=vote(high,case['choices']);gp=hp if row['gate_upgrades'] else vote(low,case['choices'])
        pairs['prediction_equal']+=int(hp==gp)
        pairs['gate_correct_high_wrong']+=int(gp==labels[case['id']] and hp!=labels[case['id']])
        pairs['gate_wrong_high_correct']+=int(gp!=labels[case['id']] and hp==labels[case['id']])
    freeze(OUT/'signal_audit.json',{'scope':'exploratory grouping, not causal mediation',
        'groups':groups,'gate_vs_all_high':pairs,'new_model_calls':0})
    freeze(OUT/'candidate_lock.json',{
        'status':'candidate fixed after opened-data exploration, not independently validated',
        'primary':'three initial low answers all valid, non-abstaining and equal: keep; otherwise request all three high reports',
        'secondary':'no-majority gate remains secondary; never select winner after validation',
        'forbidden_features':['domain ID','gold','high reports','historical correctness','question-specific rules'],
        'selector_model_calls':0,'initial_low_attempts_per_question':3,
        'high_calls_if_upgraded':3,'report_output_cap':600,
        'validation_requirements':['source-stem disjoint from all previously used questions',
            'same cases across full-low/full-high/fixed/random/length and gate',
            'freeze decision before high report collection',
            'charge initial low calls, unknown transport usage, and full-high bypass',
            'separate answer usability from AEIC completeness',
            'report corrections and harms, not just equal aggregate accuracy',
            'do not retune rule on validation outcomes'],
        'no_claims':['new game equilibrium','pure AEIC improvement','lower total tokens','independent test success'],
        'policy_code_sha256':hashlib.sha256((p.ROOT/'experiments/mechanism/replay_low_cost_router.py').read_bytes()).hexdigest()})
    tests=subprocess.run([sys.executable,'-m','unittest','experiments.tests.test_low_cost_router',
        'experiments.tests.test_resource_allocation_probe','experiments.tests.test_upgrade_pilot',
        'experiments.tests.test_upgrade_v3'],cwd=p.ROOT,text=True,capture_output=True)
    count=re.search(r'Ran (\d+) tests',tests.stderr)
    if tests.returncode or not count:raise ValueError('tests failed: '+tests.stdout+tests.stderr)
    diff=subprocess.run(['git','diff','--check'],cwd=p.ROOT,text=True,capture_output=True)
    if diff.returncode:raise ValueError('diff check failed: '+diff.stdout+diff.stderr)
    files=[p.ROOT/'experiments/mechanism/replay_low_cost_router.py',
           p.ROOT/'experiments/mechanism/audit_low_cost_router.py',
           p.ROOT/'experiments/tests/test_low_cost_router.py',*OUT.glob('*.json')]
    freeze(OUT/'acceptance.json',{'status':'OFFLINE_ROUTING_REPLAY_VERIFIED_NOT_INDEPENDENT_EFFICACY',
        'tests_passed':int(count[1]),'original_frozen_files_verified':len(frozen_hashes()),
        'new_api_calls':0,'git_diff_check_passed':True,
        'snapshot_sha256':{str(f.relative_to(p.ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
                           for f in files if f.name!='acceptance.json'}})
    print(json.dumps({'groups':groups,'gate_vs_allhigh':pairs,'tests_passed':int(count[1]),'new_api_calls':0},indent=2))


if __name__=='__main__':audit()
