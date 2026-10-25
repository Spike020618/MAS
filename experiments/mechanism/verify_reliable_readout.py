"""Offline source and execution acceptance for the minimal standard baseline."""
import argparse
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.mechanism import run_reliable_readout as run
from experiments.mechanism.run_upgrade_pilot import CallJournal,read_rows,freeze


def native():
    from experiments.mechanism.prepare_routing_holdout import PINS
    from experiments.mechanism.prepare_upgrade_p2 import pool,native_mc
    from experiments.mechanism.prepare_upgrade_pilot import canonical
    dep=Path('/private/tmp/mas_p2_dependencies')
    if dep.is_dir():sys.path.insert(0,str(dep))
    root=run.design.p.ROOT
    args=SimpleNamespace(strategyqa=root/'experiments/results/mechanism/stackelberg/locked_source_game_test_v1/official_train.json',
                         truthfulqa=root/'experiments/data/raw/upgrade_p2_v1/truthfulqa_mc_authors.json',
                         mmlu=root/'experiments/data/raw/upgrade_p2_v1/mmlu_test.parquet')
    for d,h in PINS.items():assert run.design.fingerprint(getattr(args,d))==h
    source_path,args.mmlu=args.mmlu,None
    rows,_=pool(args)
    import pyarrow.parquet as pq
    for r in pq.read_table(source_path).to_pylist():
        if len(set(r['choices']))==len(r['choices']):rows.append(native_mc('mmlu',run.design.p.digest([r['subject'],r['question']])[:24],r['question'],list(r['choices']),int(r['answer'])))
    expected={r['domain']+':'+r['source_id']:r for r in rows}
    cases=json.loads((run.OUT/'public_cases.json').read_text());labels={r['id']:r['label'] for r in json.loads((run.DATA/'private_labels.json').read_text())}
    seen=set()
    for c in cases:
        e=expected[c['id']]
        assert c['question']==e['question'] and c['choices']==e['choices'] and labels[c['id']]==e['label']
        stem=canonical(e['base_question']);assert stem not in seen;seen.add(stem)
    freeze(run.OUT/'source_verification.json',{'n':len(cases),'label_and_rendered_options_match_original':True,
           'source_pins':PINS,'scope':'data integrity, not independent factual adjudication','new_model_calls':0})
    print('native18 labels and rendered choices verified')


def verify():
    from experiments.mechanism.complete_legacy_gain_audit import frozen_hashes
    from experiments.mechanism.complete_budgeted_information import verify_prior_snapshots
    root=run.design.p.ROOT;protocol=json.loads((run.OUT/'protocol.json').read_text())
    assert run.design.fingerprint(Path(run.__file__))==protocol['code_sha256']
    cal=json.loads((run.OUT/'calibration.json').read_text())
    assert run.design.fingerprint(Path(run.calibrate.__code__.co_filename))==cal['code_sha256']
    for rel,h in cal['source_sha256'].items():assert run.design.fingerprint(root/rel)==h
    cases=json.loads((run.OUT/'public_cases.json').read_text());outputs=json.loads((run.OUT/'sealed_outputs.json').read_text())
    assert len(cases)==18 and not {c['id'] for c in cases}&set(cal['calibration_ids'])
    for name,h in json.loads((run.DATA/'provenance.json').read_text())['files'].items():assert run.design.fingerprint(run.DATA/name)==h
    before=frozen_hashes();prior=verify_prior_snapshots()
    def forbidden(_):raise AssertionError('offline verification forbids API')
    with patch('socket.socket.connect',side_effect=AssertionError('offline')),patch('socket.getaddrinfo',side_effect=AssertionError('offline')):
        journal=CallJournal(run.OUT,forbidden,180)
        for c in cases:
            for slot in ['D0','D1','D2','G0','T0']:
                if slot=='T0':req=run.thinking.request(c,'enabled')
                else:
                    b='zhipu_flash' if slot=='G0' else 'deepseek'
                    req=run.design.request(c,b,protocol['models'][b],int(slot[1]))
                row=journal.call(c['id']+'/'+slot,req)
                if row['response']['http_status']==429 or row['response']['http_status']>=500:row=journal.call(c['id']+'/'+slot+'/retry1',req)
                assert run.parse_row(row,c['choices'])==outputs[c['id']][slot]
        recomputed=run.analyze()
        names=['test_reliable_readout','test_reliable_readout_runner','test_two_backend_complementarity','test_information_complementarity',
               'test_simple_readout_crossfit','test_thinking_control','test_complementarity_extraction','test_premise_review','test_premise_runner',
               'test_interface_verdict_audit','test_legacy_gain_attribution']
        log=io.StringIO();tested=unittest.TextTestRunner(stream=log).run(unittest.defaultTestLoader.loadTestsFromNames(['experiments.tests.'+n for n in names]))
        assert tested.wasSuccessful(),log.getvalue()
    assert before==frozen_hashes()
    raw=read_rows(run.OUT/'responses.jsonl');known=[];unknown=[]
    for r in raw:
        usage=json.loads(r['response']['body']).get('usage',{}).get('total_tokens')
        if type(usage)is int:known.append(usage)
        else:unknown.append(r['key'])
    files=[p for p in run.OUT.rglob('*') if p.is_file() and p.name not in ('run.lock','acceptance.json')]
    files += [Path(__file__),Path(run.__file__),Path(run.calibrate.__code__.co_filename),
              root/'docs/research/研究主张与最小成果收口_2026-10-24.md',root/'docs/research/最小可靠性流程_独立验证结果_2026-10-24.md']
    acceptance={'status':'baseline_implemented_and_independently_tested_no_superiority_observed','new_model_task_calls':90,
       'actual_http_attempts':len(read_rows(run.OUT/'attempts.jsonl')),'known_token_lower_bound':sum(known),'unknown_usage_keys':unknown,
       'tests_passed':tested.testsRun,'frozen_original_files_verified':len(before),'prior_snapshots_verified':len(prior),
       'new_calls_during_verification':0,'sha256':{str(p.relative_to(root)):run.design.fingerprint(p) for p in files}}
    freeze(run.OUT/'acceptance.json',acceptance)
    print(json.dumps({k:v for k,v in acceptance.items() if k!='sha256'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--native',action='store_true');args=p.parse_args()
    if args.native:native()
    else:verify()
