"""Offline source/journal/test acceptance. Does not certify scientific novelty."""
import argparse
from datetime import datetime,timezone
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments.mechanism import study_two_backend_complementarity as main
from experiments.mechanism import study_explicit_thinking_control as mode
from experiments.mechanism import confirm_thinking_control as confirm
from experiments.mechanism.run_upgrade_pilot import CallJournal,read_rows,freeze
from experiments.mechanism.complete_legacy_gain_audit import frozen_hashes
from experiments.mechanism.complete_budgeted_information import verify_prior_snapshots

ROOT=main.design.p.ROOT
OUT=ROOT/'experiments/results/audit/complementarity_final_2026-10-24'
START=datetime.fromisoformat('2026-10-23T18:19:49+00:00')


def verify():
    original=frozen_hashes();old=verify_prior_snapshots()
    directories=[main.design.OUT,main.design.OUT.with_name('information_complementarity_v2'),main.OUT,mode.OUT,confirm.OUT]
    ledger=[];total_known=0;unknown=0;keys=0
    def forbidden(_):raise AssertionError('offline acceptance forbids API')
    with patch('socket.socket.connect',side_effect=AssertionError('offline')),patch('socket.getaddrinfo',side_effect=AssertionError('offline')):
        for directory in directories:
            rows=read_rows(directory/'responses.jsonl');attempts=read_rows(directory/'attempts.jsonl')
            assert len({r['key'] for r in rows})==len(rows)==len(attempts)
            journal=CallJournal(directory,forbidden,1000);known=0;missing=[]
            for r in rows:
                assert journal.call(r['key'],r['request'])==r
                b=json.loads(r['response']['body']);u=b.get('usage',{})
                if all(type(u.get(k)) is int for k in ('prompt_tokens','completion_tokens','total_tokens')):
                    assert u['total_tokens']==u['prompt_tokens']+u['completion_tokens']
                    known+=u['total_tokens']
                else:missing.append(r['key'])
            ledger.append({'directory':str(directory.relative_to(ROOT)),'http_attempts':len(rows),
                'known_token_lower_bound':known,'unknown_usage_keys':missing,
                'http_statuses':{str(s):sum(r['response']['http_status']==s for r in rows) for s in {r['response']['http_status'] for r in rows}},
                'sum_request_wall_seconds':sum(r['wall_seconds'] for r in rows)})
            total_known+=known;unknown+=len(missing);keys+=len(rows)
        for directory in (main.OUT,mode.OUT,confirm.OUT):
            prot=json.loads((directory/'protocol.json').read_text())
            if isinstance(prot['code_sha256'],dict):
                for rel,h in prot['code_sha256'].items():assert main.design.fingerprint(ROOT/rel)==h
            else:assert main.design.fingerprint(Path(mode.__file__) if directory==mode.OUT else Path(confirm.__file__))==prot['code_sha256']
        for data in (main.OUT,confirm.DATA):
            prov=json.loads((data/'provenance.json').read_text())
            for name,h in prov['files'].items():assert main.design.fingerprint(data/name)==h
        first=json.loads((main.OUT/'public_cases.json').read_text());second=json.loads((confirm.DATA/'public_cases.json').read_text())
        assert not {r['id'] for r in first}&{r['id'] for r in second}
        assert len(first)==len(second)==36
        records=read_rows(main.OUT/'responses.jsonl')
        slots=json.loads((main.OUT/'sealed_outputs.json').read_text())
        protocol=json.loads((main.OUT/'protocol.json').read_text())
        journal=CallJournal(main.OUT,forbidden,436)
        for case in first:
            for b in main.BACKENDS:
                for i in range(3):
                    req=main.design.request(case,b,protocol['models'][b],i);key=case['id']+'/'+b+'/'+str(i)
                    row=journal.call(key,req)
                    if row['response']['http_status']==429 or row['response']['http_status']>=500:row=journal.call(key+'/retry1',req)
                    assert main.parse_row(row,case['choices'])==slots[case['id']][b+'/'+str(i)]
        labels={r['id']:r['label'] for r in json.loads((main.OUT/'private_labels.json').read_text())}
        result=json.loads((main.OUT/'result.json').read_text())
        for split in ['all','calibration','evaluation']:
            recomputed=main.summarize([c for c in first if split=='all' or c['split']==split],slots,labels)
            for k,v in recomputed.items():assert result['results'][split][k]==v
        for directory,cs,data in [(mode.OUT,first,main.OUT),(confirm.OUT,second,confirm.DATA)]:
            sealed=json.loads((directory/'sealed_outputs.json').read_text())
            labels_mode={r['id']:r['label'] for r in json.loads((data/'private_labels.json').read_text())}
            recorded=json.loads((directory/'result.json').read_text())
            journal_mode=CallJournal(directory,forbidden,144)
            for case in cs:
                for arm in mode.MODES:
                    key=case['id']+'/'+arm;req=mode.request(case,arm)
                    raw=journal_mode.call(key,req)
                    if raw['response']['http_status']==429 or raw['response']['http_status']>=500:raw=journal_mode.call(key+'/retry1',req)
                    parsed=main.parse_row(raw,case['choices'])
                    assert all(sealed[case['id']][arm][k]==v for k,v in parsed.items())
            for arm in mode.MODES:
                correct=sum(sealed[c['id']][arm]['answer']==labels_mode[c['id']] for c in cs)
                expected=recorded['results']['all'][arm]['correct'] if directory==mode.OUT else recorded['modes'][arm]['correct']
                assert correct==expected
        gate=json.loads((main.OUT/'readout_followup/gate.json').read_text())
        assert gate['passed'] is False
        assert not (main.OUT/'readout_followup/attempts.jsonl').exists()
        names=['test_two_backend_complementarity','test_information_complementarity','test_simple_readout_crossfit',
               'test_thinking_control','test_complementarity_extraction','test_premise_review','test_premise_runner',
               'test_interface_verdict_audit','test_legacy_gain_attribution']
        log=io.StringIO();tested=unittest.TextTestRunner(stream=log).run(unittest.defaultTestLoader.loadTestsFromNames(['experiments.tests.'+n for n in names]))
        assert tested.wasSuccessful(),log.getvalue()
    assert frozen_hashes()==original
    OUT.mkdir(parents=True,exist_ok=True)
    ledger_result={'new_model_task_calls':360,'all_http_attempts_including_fixtures_and_failures':keys,
                  'known_tokens_lower_bound':total_known,'responses_with_unknown_usage':unknown,'runs':ledger,
                  'money_cost':'not inferred: no verified billing reconciliation','new_calls_during_verification':0}
    freeze(OUT/'cost_ledger.json',ledger_result)
    return {'tests_passed':tested.testsRun,'test_log':log.getvalue(),'frozen_original_files_verified':len(original),
            'prior_snapshots_verified':len(old),'cost_ledger':ledger_result,'source_groups_verified':72,
            'source_verification_sha256':main.design.fingerprint(main.OUT/'source_verification.json')}


def complete():
    verified=verify()
    target=OUT/'acceptance.json'
    if target.exists():
        existing=json.loads(target.read_text())
        assert existing['minimum_30_minutes_met'] and existing['elapsed_seconds']>=1800
        for rel,h in existing['sha256'].items():assert main.design.fingerprint(ROOT/rel)==h
        print('existing frozen acceptance reverified; no new API calls')
        return
    now=datetime.now(timezone.utc);elapsed=(now-START).total_seconds()
    if elapsed<1800:raise RuntimeError('minimum research duration not yet met; do not claim final acceptance')
    dirs=[main.design.OUT,main.design.OUT.with_name('information_complementarity_v2'),main.OUT,mode.OUT,confirm.OUT,confirm.DATA,
          ROOT/'experiments/results/audit/correct_minority_2026-10-24',
          ROOT/'experiments/results/audit/simple_readout_crossfit_2026-10-24']
    files=[p for d in dirs for p in d.rglob('*') if p.is_file() and p.name!='run.lock']
    files += [OUT/'cost_ledger.json',OUT/'exhaustive_accounting.json']
    docs=list((ROOT/'docs/research').glob('独立信息互补*2026-10-24.md'))
    code=[Path(__file__)]+[ROOT/'experiments/mechanism'/name for name in
        ['study_information_complementarity.py','run_complementarity_v2.py','study_two_backend_complementarity.py',
         'study_explicit_thinking_control.py','confirm_thinking_control.py','complementarity_readout_gate.py',
         'audit_complementarity_extraction.py','audit_historical_correct_minority.py','audit_simple_readout_crossfit.py',
         'audit_readout_fold_stability.py','bootstrap_simple_readout.py','verify_complementarity_sources.py']]
    freeze(OUT/'acceptance.json',{'status':'research_execution_complete_not_new_mechanism_validated',
        'started_utc':START.isoformat(),'completed_utc':now.isoformat(),'elapsed_seconds':elapsed,
        'minimum_30_minutes_met':True,**verified,
        'sha256':{str(p.relative_to(ROOT)):main.design.fingerprint(p) for p in files+docs+code}})
    print(json.dumps({'elapsed_minutes':elapsed/60,'tests_passed':verified['tests_passed'],
                      'http_attempts':verified['cost_ledger']['all_http_attempts_including_fixtures_and_failures']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true');args=parser.parse_args()
    if args.final:complete()
    else:
        v=verify();print('offline checks passed',v['tests_passed'],'tests; final duration not yet asserted')
