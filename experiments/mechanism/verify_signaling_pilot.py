"""Offline source, shared-pool, protocol and legacy-integrity acceptance."""
import argparse
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

from experiments.mechanism import run_signaling_pilot as run
from experiments.mechanism.run_upgrade_pilot import CallJournal,read_rows,freeze


def native():
    if not (run.OUT/'sealed_readouts.json').exists():
        raise ValueError('seal readouts before opening source labels for verification')
    from experiments.mechanism.prepare_routing_holdout import PINS
    from experiments.mechanism.prepare_upgrade_p2 import pool,native_mc
    from experiments.mechanism.prepare_upgrade_pilot import canonical
    dep=Path('/private/tmp/mas_p2_dependencies')
    if dep.is_dir():sys.path.insert(0,str(dep))
    root=run.ROOT
    args=SimpleNamespace(strategyqa=root/'experiments/results/mechanism/stackelberg/locked_source_game_test_v1/official_train.json',
                         truthfulqa=root/'experiments/data/raw/upgrade_p2_v1/truthfulqa_mc_authors.json',
                         mmlu=root/'experiments/data/raw/upgrade_p2_v1/mmlu_test.parquet')
    for d,h in PINS.items():assert run.fingerprint(getattr(args,d))==h
    mmlu,args.mmlu=args.mmlu,None
    rows,_=pool(args)
    import pyarrow.parquet as pq
    from experiments.mechanism.probe_resource_allocation import digest
    for r in pq.read_table(mmlu).to_pylist():
        if len(set(r['choices']))==len(r['choices']):
            rows.append(native_mc('mmlu',digest([r['subject'],r['question']])[:24],r['question'],list(r['choices']),int(r['answer'])))
    expected={r['domain']+':'+r['source_id']:r for r in rows}
    cases=json.loads((run.OUT/'public_cases.json').read_text())
    labels={r['id']:r['label'] for r in json.loads((run.SOURCE/'private_labels.json').read_text())}
    seen=set()
    for c in cases:
        e=expected[c['id']]
        assert c['question']==e['question'] and c['choices']==e['choices'] and labels[c['id']]==e['label']
        stem=canonical(e['base_question']);assert stem not in seen;seen.add(stem)
    freeze(run.OUT/'source_verification.json',{'n':len(cases),'source_pins':PINS,
        'labels_and_rendered_choices_match_original':True,'unique_source_stems':len(seen),
        'scope':'integrity only, not independent factual adjudication or untouched benchmark provenance','new_api_calls':0})
    print('native12 labels and rendered choices verified')


def verify():
    from experiments.mechanism.complete_legacy_gain_audit import frozen_hashes
    from experiments.mechanism.complete_budgeted_information import verify_prior_snapshots
    original=frozen_hashes();prior=verify_prior_snapshots()
    initial=run.ROOT/'experiments/results/audit/signaling_foundation_v1/acceptance.json'
    for rel,h in json.loads(initial.read_text())['sha256'].items():assert run.fingerprint(run.ROOT/rel)==h,rel
    protocol=json.loads((run.OUT/'protocol.json').read_text())
    for rel,h in protocol['code_sha256'].items():assert run.fingerprint(run.ROOT/rel)==h,rel
    selection=json.loads((run.OUT/'selection.json').read_text())
    for rel,h in selection['source_hashes'].items():assert run.fingerprint(run.ROOT/rel)==h,rel
    assert run.fingerprint(run.OUT/'public_cases.json')==protocol['public_sha256']==selection['public_sha256']
    cases=json.loads((run.OUT/'public_cases.json').read_text());sealed=json.loads((run.OUT/'sealed_outputs.json').read_text())
    assert len(cases)==12 and len({c['id'] for c in cases})==12
    assert json.loads((run.OUT/'source_verification.json').read_text())['n']==12
    expected_keys=set()
    def forbidden(_):raise AssertionError('offline replay cannot call API')
    with patch('socket.socket.connect',side_effect=AssertionError('offline')),patch('socket.getaddrinfo',side_effect=AssertionError('offline')):
        journal=CallJournal(run.OUT,forbidden,protocol['attempt_limit'])
        for c in cases:
            for slot in ['g_positive','g_negative','g_positive_reverse','g_negative_reverse']+['d/'+x for x in c['choices']]:
                key=c['id']+'/'+slot;expected_keys.add(key)
                row=journal.call(key,run.request(c,slot,protocol['model_requested']))
                assert run.parsed(row,c,slot)==sealed[c['id']][slot]
        assert {r['key'] for r in read_rows(run.OUT/'responses.jsonl')}==expected_keys
        assert {r['key'] for r in read_rows(run.OUT/'attempts.jsonl')}==expected_keys
        recomputed=run.analyze()
        names=['test_two_backend_complementarity','test_information_complementarity','test_simple_readout_crossfit',
            'test_thinking_control','test_complementarity_extraction','test_premise_review','test_premise_runner',
            'test_interface_verdict_audit','test_legacy_gain_attribution','test_reliable_readout','test_reliable_readout_runner',
            'test_signaling_game','test_signaling_interface','test_token_signaling_adapter','test_scored_signaling']
        log=io.StringIO();tested=unittest.TextTestRunner(stream=log).run(unittest.defaultTestLoader.loadTestsFromNames(
            ['experiments.tests.'+name for name in names]))
        assert tested.wasSuccessful(),log.getvalue()
    assert frozen_hashes()==original
    files=[p for p in run.OUT.iterdir() if p.is_file() and p.name not in ('run.lock','acceptance.json')]
    files += [Path(__file__),Path(run.__file__),run.ROOT/'experiments/mechanism/diagnose_signaling_pilot.py',
              run.ROOT/'experiments/core/mechanism/scored_signaling.py',
              run.ROOT/'experiments/tests/test_scored_signaling.py',
              run.ROOT/'docs/research/语义共识博弈_P2同模型小批量结果_2026-10-24.md']
    acceptance={'status':'developmental_pool_verified_not_confirmatory_result','tests_passed':tested.testsRun,
        'frozen_original_files_verified':len(original),'prior_snapshots_verified':len(prior),
        'foundation_acceptance_verified':run.fingerprint(initial),'new_api_calls_during_acceptance':0,
        'new_benchmark_scoring_calls':len(expected_keys),
        'collection_known_tokens':recomputed['known_tokens_lower_bound'],
        'sha256':{str(p.relative_to(run.ROOT)):run.fingerprint(p) for p in files}}
    freeze(run.OUT/'acceptance.json',acceptance)
    print(json.dumps({k:v for k,v in acceptance.items() if k!='sha256'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--native',action='store_true');args=parser.parse_args()
    if args.native:native()
    else:verify()
