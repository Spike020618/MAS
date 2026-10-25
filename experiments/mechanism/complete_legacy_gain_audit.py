"""Reproduce the bounded legacy attribution audit without network/model calls.

Only generated audit outputs/logs are written. Historical evidence and protected
protocols remain untouched. The manifest is a local integrity snapshot, not an
independent preregistration or factual-quality certificate.
"""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'experiments/results/audit/legacy_gain_attribution_2026-10-22'
MODULES = ['audit_legacy_source_groups','attribute_legacy_gain','audit_legacy_conclusion_parsing',
           'audit_legacy_controls','audit_legacy_prompt_boundary','audit_legacy_measurement',
           'extend_legacy_sensitivity','audit_legacy_aggregation','audit_legacy_explanatory_controls',
           'audit_legacy_route_pairs','audit_legacy_theory_bridge']
TESTS = ['experiments.tests.test_legacy_gain_attribution','experiments.tests.test_resource_identification',
         'experiments.tests.test_contract_game_alignment','experiments.tests.test_research_closeout',
         'experiments.tests.test_resource_execution','experiments.tests.test_resource_target_probe',
         'experiments.tests.test_deliberation_stagnation']


def fingerprint(path):return sha256(path.read_bytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()


def frozen_hashes():
    checked = {}
    for provenance in sorted((ROOT/'experiments/results/mechanism/frozen').glob('*/provenance.json')):
        if provenance.parent.name.endswith('.recomputed'):continue
        for name,expected in json.loads(provenance.read_text())['files'].items():
            path=provenance.parent/name
            if fingerprint(path)!=expected:raise ValueError('frozen evidence changed: '+str(path))
            checked[str(path.relative_to(ROOT))]=expected
    if len(checked)!=169:raise ValueError('unexpected frozen evidence coverage')
    return checked


def refresh_manifest():
    outputs = {str(p.relative_to(ROOT)):fingerprint(p) for p in sorted(OUT.rglob('*')) if p.is_file()
               and p.name!='reproducibility_manifest.json'}
    scripts = {str((ROOT/'experiments/mechanism'/ (m+'.py')).relative_to(ROOT)):
               fingerprint(ROOT/'experiments/mechanism'/(m+'.py')) for m in MODULES}
    scripts[str(Path(__file__).resolve().relative_to(ROOT))]=fingerprint(Path(__file__).resolve())
    tests=ROOT/'experiments/tests/test_legacy_gain_attribution.py'
    scripts[str(tests.relative_to(ROOT))]=fingerprint(tests)
    docs={str(p.relative_to(ROOT)):fingerprint(p) for p in sorted((ROOT/'docs/research').glob('旧方案收益归因*2026-10-22.md'))}
    sources={}
    for p in sorted(OUT.glob('*.json')):
        if p.name=='reproducibility_manifest.json':continue
        r=json.loads(p.read_text())
        for field in ['source_hashes','inputs_sha256','inputs_and_code_sha256']:
            for rel,expected in r.get(field,{}).items():
                path=ROOT/rel
                if not path.is_file() or fingerprint(path)!=expected:
                    raise ValueError('analysis input commitment stale: '+rel+' from '+p.name)
                sources[rel]=expected
    manifest={'scope':'local reproducibility snapshot, not preregistration or independent truth assessment',
              'generated_at_utc':utc(),'outputs_and_logs':outputs,'analysis_code':scripts,
              'research_deliverables':docs,'verified_analysis_inputs':sources,
              'frozen_original_files':frozen_hashes(),'paid_model_calls_added':0}
    (OUT/'reproducibility_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return manifest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dependency-path',type=Path)
    parser.add_argument('--refresh-manifest-only',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.refresh_manifest_only:
        m=refresh_manifest();print(json.dumps({'manifest_refreshed':True,'outputs':len(m['outputs_and_logs'])}));return
    dependency=args.dependency_path
    if dependency is None and Path('/private/tmp/mas_p2_dependencies').is_dir():
        dependency=Path('/private/tmp/mas_p2_dependencies')
    if dependency:sys.path.insert(0,str(dependency))
    started=utc(); tick=time.monotonic(); stages=[]; tests=[]
    protected_before=frozen_hashes()
    logs=OUT/'verification_logs';logs.mkdir(exist_ok=True)
    acceptance={'status':'RUNNING','started_at_utc':started,'paid_model_calls_added':0,
                'budget_window_start_utc':'2026-10-21T20:15:00+00:00',
                'budget_window_deadline_utc':'2026-10-22T02:15:00+00:00',
                'scope':'research attribution audit only; not new method efficacy completion'}
    try:
        for module in MODULES:
            stdout,stderr=io.StringIO(),io.StringIO();before=time.monotonic()
            # Enforce offline analysis, including accidental HTTP/DNS attempts.
            with patch('socket.socket.connect',side_effect=RuntimeError('offline audit forbids network')), \
                 patch('socket.socket.connect_ex',side_effect=RuntimeError('offline audit forbids network')), \
                 patch('socket.getaddrinfo',side_effect=RuntimeError('offline audit forbids DNS')), \
                 redirect_stdout(stdout),redirect_stderr(stderr):
                runpy.run_module('experiments.mechanism.'+module,run_name='__main__')
            (logs/(module+'.stdout.txt')).write_text(stdout.getvalue())
            (logs/(module+'.stderr.txt')).write_text(stderr.getvalue())
            stages.append({'module':module,'passed':True,'seconds':time.monotonic()-before})
        for name,command in [('upgrade_tests',[sys.executable,'-m','unittest','discover','-s','experiments/tests','-p','test_upgrade*.py']),
                             ('legacy_and_related_tests',[sys.executable,'-m','unittest',*TESTS])]:
            process=subprocess.run(command,cwd=ROOT,text=True,capture_output=True)
            (logs/(name+'.txt')).write_text(process.stdout+'\n'+process.stderr)
            found=re.search(r'Ran (\d+) tests?',process.stderr)
            tests.append({'group':name,'passed':process.returncode==0,'tests':int(found[1]) if found else None})
            if process.returncode or found is None:raise ValueError(name+' failed; inspect audit log')
        diff=subprocess.run(['git','diff','--check'],cwd=ROOT,text=True,capture_output=True)
        (logs/'git_diff_check.txt').write_text(diff.stdout+diff.stderr)
        if diff.returncode:raise ValueError('git diff --check failed')
        if frozen_hashes()!=protected_before:raise ValueError('frozen evidence changed during audit')
        from experiments.mechanism.calibrate_upgrade_v2 import audit_stage
        root=ROOT/'experiments/results/upgrade_game'
        protected_stages={s:audit_stage(root/('p2_'+s+'_v2')) for s in ['calibration','development']}
        acceptance.update(status='PASS',analysis_stages=stages,test_groups=tests,
            total_tests=sum(g['tests'] for g in tests),frozen_files_verified=len(protected_before),
            frozen_evidence_unchanged=True,git_diff_check_passed=True,
            analysis_network_calls_forbidden=True,protected_P2_stage_audits=protected_stages,
            new_formal_test_started=False,remaining_scientific_gap='no identified pure AEIC/incentive increment; no unified effective strict-game method')
    except Exception as exc:
        acceptance.update(status='FAIL',error=type(exc).__name__+': '+str(exc),analysis_stages=stages,test_groups=tests)
        raise
    finally:
        acceptance['finished_at_utc']=utc();acceptance['runner_wall_seconds']=time.monotonic()-tick
        acceptance['elapsed_budget_window_minutes']=(datetime.now(timezone.utc)-datetime.fromisoformat(acceptance['budget_window_start_utc'])).total_seconds()/60
        acceptance['timing_note']='elapsed since approximate first-phase start, NOT continuous active work or six hours consumed'
        (OUT/'acceptance.json').write_text(json.dumps(acceptance,ensure_ascii=False,indent=2)+'\n')
    manifest=refresh_manifest()
    print(json.dumps({'status':acceptance['status'],'tests':acceptance['total_tests'],
        'frozen_files':acceptance['frozen_files_verified'],'outputs':len(manifest['outputs_and_logs']),
        'paid_model_calls_added':0,'elapsed_budget_window_minutes':acceptance['elapsed_budget_window_minutes']},ensure_ascii=False))


if __name__=='__main__':main()
