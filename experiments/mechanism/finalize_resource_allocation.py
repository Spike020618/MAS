"""Offline engineering acceptance, explicitly not research efficacy acceptance."""
import hashlib
import json
import re
import subprocess
import sys

from experiments.mechanism import probe_resource_allocation as p
from experiments.mechanism.finish_resource_allocation import bounds
from experiments.mechanism.sensitivity_resource_allocation import main as sensitivity
from experiments.mechanism.complete_legacy_gain_audit import frozen_hashes
from experiments.mechanism.run_upgrade_pilot import freeze,read_rows


if __name__=='__main__':
    result=bounds();sensitivity()
    tests=subprocess.run([sys.executable,'-m','unittest',
        'experiments.tests.test_resource_allocation_probe','experiments.tests.test_upgrade_pilot',
        'experiments.tests.test_upgrade_v3'],cwd=p.ROOT,capture_output=True,text=True)
    found=re.search(r'Ran (\d+) tests',tests.stderr)
    if tests.returncode or not found:raise ValueError('tests failed: '+tests.stdout+tests.stderr)
    diff=subprocess.run(['git','diff','--check'],cwd=p.ROOT,capture_output=True,text=True)
    if diff.returncode:raise ValueError('diff failed: '+diff.stdout+diff.stderr)
    protected=frozen_hashes()
    old=p.OUT.parent/'probe_v1'
    original=json.loads((old/'protocol.json').read_text())
    if original['code_hash']!=p.digest((old/'source.py').read_text()):raise ValueError('v1 source changed')
    amendment=json.loads((p.OUT/'connection_amendment.json').read_text())
    source=p.ROOT/'experiments/mechanism/finish_resource_allocation.py'
    if amendment['completion_code_hash']!=p.digest(source.read_text()):raise ValueError('completion source changed')
    files=list(p.OUT.glob('*.json'))+list(p.OUT.glob('*.jsonl'))
    files += [p.ROOT/'experiments/mechanism'/name for name in [
        'probe_resource_allocation.py','finish_resource_allocation.py',
        'sensitivity_resource_allocation.py','finalize_resource_allocation.py']]
    files += [p.ROOT/'docs/research/资源分配小验证_2026-10-22.md',
              p.ROOT/'docs/research/资源分配小验证_结果_2026-10-22.md',
              p.ROOT/'experiments/tests/test_resource_allocation_probe.py']
    acceptance={'status':'BOUNDED_DIAGNOSTIC_VERIFIED_NOT_EFFICACY_ACCEPTANCE',
        'tests_passed':int(found[1]),'git_diff_check_passed':True,
        'original_frozen_files_verified':len(protected),
        'new_http_attempts':len(read_rows(old/'attempts.jsonl'))+result['attempts'],
        'successful_responses':result['successful_responses'],
        'unresolved_transport_outcomes':len(result['unresolved_slots']),
        'unknown_usage_not_zero_filled':True,'paid_calls_during_verification':0,
        'independent_holdout':False,'research_goal_achieved':False,
        'snapshot_sha256':{str(f.relative_to(p.ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
                           for f in files if f.name!='diagnostic_acceptance.json'}}
    freeze(p.OUT/'diagnostic_acceptance.json',acceptance)
    print(json.dumps({k:v for k,v in acceptance.items() if k!='snapshot_sha256'},indent=2))
