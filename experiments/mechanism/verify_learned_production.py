"""Offline execution audit and frozen-evidence preservation for production v1."""
from fractions import Fraction
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments.mechanism import run_learned_production as r
from experiments.core.mechanism import learned_production_game as g
from experiments.mechanism.run_upgrade_pilot import freeze,read_rows
from experiments.mechanism.complete_legacy_gain_audit import frozen_hashes
from experiments.mechanism.complete_budgeted_information import verify_prior_snapshots


def main():
    original=frozen_hashes();prior=verify_prior_snapshots();out=r.OUT
    protocol=json.loads((out/'protocol.json').read_text())
    for rel,value in protocol['code_sha256'].items():assert r.h(r.ROOT/rel)==value,rel
    for name,value in protocol['data_sha256'].items():assert r.h(out/name)==value,name
    assert r.h(r.ROOT/'docs/research/可学习信息生产协调_锁定方案_2026-10-25.md')==protocol['design_sha256']
    prov=json.loads((out/'source_data/provenance.json').read_text())
    for name,value in prov['files'].items():assert r.h(out/'source_data'/name)==value
    policy=json.loads((out/'frozen_policy.json').read_text());train=json.loads((out/'train_public.json').read_text())
    outputs=json.loads((out/'train_sealed_outputs.json').read_text());labels=json.loads((out/'train_labels.json').read_text())
    expected=g.fit([{'slots':outputs[c['id']],'label':labels[c['id']],
                    'context':g.context({k:outputs[c['id']][k] for k in ('fast0','fast1','fast2')})} for c in train])
    assert policy==expected
    for state in policy['contexts'].values():
        t=state['table'];profile=state['game']['profile']
        for p in g.PROFILES:
            assert g.field(t,state['centralized_erm'],'potential')>=g.field(t,p,'potential')
            for i in range(2):
                q=list(p);q[i]=1-q[i]
                assert g.field(t,q,'utilities',i)-g.field(t,p,'utilities',i)==g.field(t,q,'potential')-g.field(t,p,'potential')
        assert state['game']['empirical_regrets']==['0','0']
    allrows={}
    for split in ('train','test'):
        cases=json.loads((out/(split+'_public.json')).read_text());rows=[]
        for case in cases:
            directory=out/split/r.sha256(case['id'].encode()).hexdigest()
            rr=read_rows(directory/'responses.jsonl');aa=read_rows(directory/'attempts.jsonl')
            assert len(rr)==len(aa)==5
            assert [x['key'] for x in rr[:3]]==['fast0','fast1','fast2']
            if split=='test':
                decision=json.loads((directory/'decision.json').read_text())
                when=json.loads((directory/'decision_time.json').read_text())['utc']
                assert decision['policy_sha256']==r.h(out/'frozen_policy.json')
                assert max(x['completed_utc'] for x in rr[:3])<=when<=min(x['started_utc'] for x in aa[3:])
                selected=['think'+str(i+1) for i,a in enumerate(decision['game']) if a]
                assert [x['key'] for x in rr[3:3+len(selected)]]==selected
            rows.extend(rr)
        allrows[split]=rows
    assert max(x['completed_utc'] for x in allrows['train'])<min(x['completed_utc'] for x in allrows['test'])
    names=('test_equilibrium_ranking_reference','test_equilibrium_reference_pipeline','test_signaling_game',
           'test_token_signaling_adapter','test_scored_signaling','test_verification_risk_game',
           'test_verification_verdict_sensitivity','test_learned_production_game','test_learned_production_runner')
    with (patch('socket.socket.connect',side_effect=AssertionError('offline')),
          patch('socket.getaddrinfo',side_effect=AssertionError('offline')),
          patch.object(r.api,'transport',side_effect=AssertionError('resume cannot call API'))):
        with patch('builtins.print'):
            r.execute('train');r.execute('test')
            from experiments.mechanism.diagnose_learned_production import main as diagnosis
            from experiments.mechanism.summarize_production_single_controls import main as singles
            from experiments.mechanism.audit_production_interaction import main as interaction
            diagnosis();singles();interaction()
        stream=io.StringIO()
        tested=unittest.TextTestRunner(stream=stream).run(unittest.defaultTestLoader.loadTestsFromNames(
            ['experiments.tests.'+name for name in names]))
        assert tested.wasSuccessful(),stream.getvalue()
    assert frozen_hashes()==original
    previous=r.ROOT/'experiments/results/audit/equilibrium_reference_replay_2026-10-25/acceptance.json'
    for rel,value in json.loads(previous.read_text())['sha256'].items():assert r.h(r.ROOT/rel)==value,rel
    latest=r.ROOT/'experiments/results/resource_allocation/deepseek_pikl_reserved_v1/final_acceptance.json'
    for rel,value in json.loads(latest.read_text())['sha256'].items():assert r.h(r.ROOT/rel)==value,rel
    result=json.loads((out/'result.json').read_text())
    assert result['physical_calls']==120 and result['physical_tokens']==48234
    paths=[p for p in out.rglob('*') if p.is_file() and p.name not in ('run.lock','acceptance.json')]
    paths += [Path(__file__),Path(r.__file__),Path(g.__file__),
        r.ROOT/'experiments/mechanism/diagnose_learned_production.py',
        r.ROOT/'experiments/mechanism/summarize_production_single_controls.py',
        r.ROOT/'experiments/mechanism/audit_production_interaction.py',
        r.ROOT/'experiments/tests/test_learned_production_game.py',
        r.ROOT/'experiments/tests/test_learned_production_runner.py',
        r.ROOT/'docs/research/可学习信息生产协调_锁定方案_2026-10-25.md',
        r.ROOT/'docs/research/可学习协调_三步执行结果_2026-10-25.md']
    accepted={'status':'implementation_and_small_prospective_test_complete_no_game_increment',
        'tests':tested.testsRun,'original_frozen_files':len(original),'prior_snapshots':len(prior),
        'physical_calls':120,'physical_tokens':48234,'verification_new_calls':0,
        'strict_consensus_game_full_sequence_replication':'not_completed_with_current_API',
        'ECON_replication':False,'new_effective_research_mechanism':False,
        'sha256':{str(p.relative_to(r.ROOT)):r.h(p) for p in paths}}
    freeze(out/'acceptance.json',accepted)
    print(json.dumps({k:v for k,v in accepted.items() if k!='sha256'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
