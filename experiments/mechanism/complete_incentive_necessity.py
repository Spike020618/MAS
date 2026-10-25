"""Reproduce audit under a no-network guard; preserve all frozen evidence."""
import hashlib
import json
import re
import subprocess
import sys
from unittest.mock import patch

from experiments.mechanism import audit_incentive_necessity as audit
from experiments.mechanism.audit_incentive_resource_profiles import run as resource_run
from experiments.mechanism.complete_legacy_gain_audit import frozen_hashes
from experiments.mechanism.run_upgrade_pilot import freeze


if __name__=='__main__':
    root=audit.p.ROOT
    protected=frozen_hashes()
    # Every prior per-study acceptance snapshot stays unchanged as well.
    snapshots=[]
    for directory in ['probability_routing_v1','routing_holdout_validation_v1','aeic_context_v1','dispute_review_v1']:
        path=root/'experiments/results/resource_allocation'/directory/'acceptance.json'
        value=json.loads(path.read_text())
        for rel,expected in value['snapshot_sha256'].items():
            if hashlib.sha256((root/rel).read_bytes()).hexdigest()!=expected:raise ValueError('prior study changed '+rel)
        snapshots.append(str(path.relative_to(root)))
    with patch('socket.getaddrinfo',side_effect=RuntimeError('audit forbids DNS')), \
         patch('socket.socket.connect',side_effect=RuntimeError('audit forbids network')), \
         patch('socket.socket.connect_ex',side_effect=RuntimeError('audit forbids network')):
        result=audit.run();profiles=resource_run()
    modules=['test_incentive_necessity','test_incentive_resource_profiles','test_routing_holdout',
        'test_probability_compatibility','test_probability_routing','test_dispute_review',
        'test_aeic_context_probe','test_low_cost_router','test_resource_allocation_probe',
        'test_upgrade_pilot','test_upgrade_v3']
    tests=subprocess.run([sys.executable,'-m','unittest',*['experiments.tests.'+m for m in modules]],cwd=root,text=True,capture_output=True)
    count=re.search(r'Ran (\d+) tests',tests.stderr)
    if tests.returncode or not count:raise ValueError('tests failed: '+tests.stdout+tests.stderr)
    diff=subprocess.run(['git','diff','--check'],cwd=root,text=True,capture_output=True)
    if diff.returncode:raise ValueError('diff check failed: '+diff.stdout+diff.stderr)
    if frozen_hashes()!=protected:raise ValueError('frozen evidence changed')
    files=[*audit.OUT.glob('*.json'),root/'docs/research/信息获取博弈_必要性核查_2026-10-23.md']
    files += [root/'experiments/mechanism'/name for name in ['audit_incentive_necessity.py',
        'audit_incentive_resource_profiles.py','complete_incentive_necessity.py']]
    files += [root/'experiments/tests'/name for name in ['test_incentive_necessity.py','test_incentive_resource_profiles.py']]
    acceptance={'status':'OFFLINE_NECESSITY_AUDIT_VERIFIED_NOT_EFFECTIVE_MECHANISM_ACCEPTANCE',
        'new_api_calls':0,'audit_network_forbidden':True,'tests_passed':int(count[1]),
        'original_frozen_files_verified':len(protected),'prior_study_snapshots_verified':snapshots,
        'git_diff_check_passed':True,
        'resource_profitable_harm_branches':sum(v['profitable_keep_and_majority_harm'] for v in profiles['resource_results'].values()),
        'resource_cost_bearer_is_new_assumption':True,'observed_strategic_llm_behavior':False,
        'complete_game_or_optimal_contract':False,
        'snapshot_sha256':{str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in files if f.name!='acceptance.json'}}
    freeze(audit.OUT/'acceptance.json',acceptance)
    print(json.dumps({k:v for k,v in acceptance.items() if k!='snapshot_sha256'},ensure_ascii=False,indent=2))
