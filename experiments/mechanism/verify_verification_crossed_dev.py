"""Offline acceptance for the complete3x2 DEVELOPMENT table, never efficacy proof."""
from hashlib import sha256
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments.mechanism import run_verification_crossed_dev as run
from experiments.mechanism.run_upgrade_pilot import CallJournal,read_rows,freeze
from experiments.mechanism.complete_legacy_gain_audit import frozen_hashes
from experiments.mechanism.complete_budgeted_information import verify_prior_snapshots


def verify():
    original=frozen_hashes();prior=verify_prior_snapshots()
    checked=[]
    for rel in ('experiments/results/audit/verification_risk_design_v1/acceptance.json',
                'experiments/results/resource_allocation/signaling_pilot_v1/acceptance.json'):
        a=json.loads((run.ROOT/rel).read_text())
        for p,h in a['sha256'].items():assert run.fingerprint(run.ROOT/p)==h,p
        checked.append({'path':rel,'sha256':run.fingerprint(run.ROOT/rel)})
    p=json.loads((run.OUT/'protocol.json').read_text())
    for rel,h in (p['source_hashes']|p['code_hashes']).items():assert run.fingerprint(run.ROOT/rel)==h,rel
    cases=json.loads((run.OUT/'public_cases.json').read_text());sealed=json.loads((run.OUT/'sealed_outputs.json').read_text())
    assert len(cases)==12 and sum(len(c['choices']) for c in cases)==51
    rows=read_rows(run.OUT/'responses.jsonl');attempts=read_rows(run.OUT/'attempts.jsonl')
    assert len(rows)==len(attempts)==p['planned_call_limit']==306
    raw={r['key']:r for r in rows};assert len(raw)==306
    def forbidden(_):raise AssertionError('offline journal replay cannot call API')
    names=['test_two_backend_complementarity','test_information_complementarity','test_simple_readout_crossfit',
        'test_thinking_control','test_complementarity_extraction','test_premise_review','test_premise_runner',
        'test_interface_verdict_audit','test_legacy_gain_attribution','test_reliable_readout','test_reliable_readout_runner',
        'test_signaling_game','test_signaling_interface','test_token_signaling_adapter','test_scored_signaling',
        'test_signaling_safety_audit','test_verification_risk_game','test_verification_crossed_dev',
        'test_verification_verdict_sensitivity']
    with patch('socket.socket.connect',side_effect=AssertionError('offline')),patch('socket.getaddrinfo',side_effect=AssertionError('offline')):
        journal=CallJournal(run.OUT,forbidden,306)
        for c in cases:
            for candidate in c['choices']:
                prefix=c['id']+'/'+candidate;s=sealed[c['id']][candidate]
                author=journal.call(prefix+'/author',run.request(c,candidate,'author',p['model_requested']))
                assert run.decode(author,True)==s['author']
                messages={'neutral':s['author']['text'],'pressure':run.PRESSURE+'\n\n'+s['author']['text']}
                proposal=json.loads((run.OUT/('proposal_'+sha256(prefix.encode()).hexdigest()[:24]+'.json')).read_text())
                assert proposal=={'question_id':c['id'],'claim_id':candidate,'messages':messages,'truth_access':False}
                independent=journal.call(prefix+'/independent',run.request(c,candidate,'independent',p['model_requested']))
                assert run.decode(independent)==s['independent']
                for v in run.VERIFIERS[1:]:
                    for a in run.ATTACKS:
                        row=journal.call(prefix+'/'+v+'/'+a,run.request(c,candidate,v,p['model_requested'],messages[a]))
                        assert run.decode(row)==s['votes'][v+'/'+a]
                        assert rows.index(author)<rows.index(row)
        result=run.analyze()
        from experiments.mechanism.diagnose_verification_crossed_dev import main as diagnosis
        from experiments.mechanism.verification_verdict_sensitivity import main as sensitivity
        from experiments.mechanism.audit_verification_ties import main as ties
        with patch('builtins.print'):diagnosis();sensitivity();ties()
        log=io.StringIO();tested=unittest.TextTestRunner(stream=log).run(unittest.defaultTestLoader.loadTestsFromNames(
            ['experiments.tests.'+name for name in names]))
        assert tested.wasSuccessful(),log.getvalue()
    assert frozen_hashes()==original
    paths=[x for x in run.OUT.iterdir() if x.is_file() and x.name not in ('run.lock','acceptance.json')]
    paths += [Path(__file__),Path(run.__file__),run.ROOT/'experiments/mechanism/diagnose_verification_crossed_dev.py',
              run.ROOT/'experiments/tests/test_verification_crossed_dev.py',
              run.ROOT/'experiments/mechanism/verification_verdict_sensitivity.py',
              run.ROOT/'experiments/mechanism/audit_verification_ties.py',
              run.ROOT/'experiments/tests/test_verification_verdict_sensitivity.py',
              run.ROOT/'docs/research/核验交叉开发_结果与取舍_2026-10-24.md']
    acceptance={'status':'developmental_crossed_table_verified_not_independent_effectiveness',
        'tests_passed':tested.testsRun,'original_frozen_files_verified':len(original),'prior_snapshots_verified':len(prior),
        'previous_acceptances_verified':checked,'physical_api_calls':306,'logical_judgments':306,
        'author_calls':51,'physical_verifier_calls':255,'original_questions':12,
        'known_tokens_lower_bound':result['known_tokens_lower_bound'],'new_api_calls_during_acceptance':0,
        'sha256':{str(x.relative_to(run.ROOT)):run.fingerprint(x) for x in paths}}
    freeze(run.OUT/'acceptance.json',acceptance)
    print(json.dumps({k:v for k,v in acceptance.items() if k not in ('sha256','previous_acceptances_verified')},ensure_ascii=False,indent=2))


if __name__=='__main__':verify()
