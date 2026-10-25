"""Offline P0/P1 acceptance; no evidence of model accuracy superiority implied."""
from hashlib import sha256
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments.mechanism.complete_legacy_gain_audit import frozen_hashes
from experiments.mechanism.complete_budgeted_information import verify_prior_snapshots
from experiments.mechanism.run_upgrade_pilot import CallJournal,freeze,read_rows
from experiments.mechanism.probe_signaling_interface import extract
from experiments.core.mechanism.signaling_game import from_spec

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/results/audit/signaling_foundation_v1'


def fingerprint(path):return sha256(path.read_bytes()).hexdigest()


def main():
    original=frozen_hashes();prior=verify_prior_snapshots()
    recent={}
    for rel in ('experiments/results/resource_allocation/reliable_readout_v1/acceptance.json',
                'experiments/results/audit/complementarity_final_2026-10-24/acceptance.json'):
        record=json.loads((ROOT/rel).read_text())
        for file,h in record['sha256'].items():assert fingerprint(ROOT/file)==h,file
        recent[rel]=fingerprint(ROOT/rel)
    tests=['test_two_backend_complementarity','test_information_complementarity','test_simple_readout_crossfit',
           'test_thinking_control','test_complementarity_extraction','test_premise_review','test_premise_runner',
           'test_interface_verdict_audit','test_legacy_gain_attribution','test_reliable_readout',
           'test_reliable_readout_runner','test_signaling_game','test_signaling_interface','test_token_signaling_adapter']
    with patch('socket.socket.connect',side_effect=AssertionError('offline acceptance')),\
         patch('socket.getaddrinfo',side_effect=AssertionError('offline acceptance')):
        log=io.StringIO();run=unittest.TextTestRunner(stream=log).run(unittest.defaultTestLoader.loadTestsFromNames(
            ['experiments.tests.'+name for name in tests]))
        assert run.wasSuccessful(),log.getvalue()
        attempts=responses=known=unknown=0;unresolved=[];probe_files=[]
        for version in (1,2,3):
            directory=ROOT/f'experiments/results/resource_allocation/signaling_interface_probe_v{version}'
            a=read_rows(directory/'attempts.jsonl');r=read_rows(directory/'responses.jsonl')
            attempts+=len(a);responses+=len(r)
            protocol=json.loads((directory/'protocol.json').read_text())
            assert fingerprint(ROOT/'experiments/mechanism/probe_signaling_interface.py')==protocol['code_sha256']
            if version>1:
                amendment=json.loads((directory/'fixture_amendment.json').read_text())
                assert fingerprint(ROOT/f'experiments/mechanism/probe_signaling_interface_v{version}.py')==amendment['wrapper_sha256']
            def forbidden(_):raise AssertionError('replay cannot call API')
            journal=CallJournal(directory,forbidden,8)
            for row in r:
                assert journal.call(row['key'],row['request'])==row
                assert row['request']['backend'] in ('deepseek','zhipu_flash')
                value=extract(row)['tokens']
                if type(value) is int and value>=0:known+=value
                else:unknown+=1
            pending={x['key'] for x in a}-{x['key'] for x in r}
            unresolved.extend({'directory':str(directory.relative_to(ROOT)),'key':key} for key in sorted(pending))
            if version>1:
                result=json.loads((directory/'result.json').read_text())
                assert result['http_attempts']==len(a)
                for row in r:
                    backend,slot=row['key'].split('/')
                    assert result['backends'][backend]['slots'][slot]==extract(row)
            probe_files.extend(p for p in directory.iterdir() if p.is_file() and p.name!='run.lock')
        assert (attempts,responses,known,len(unresolved))==(7,6,252,1)
        fixtures=ROOT/'experiments/results/audit/signaling_game_v1'
        cases=json.loads((fixtures/'fixtures.json').read_text())
        assert len(cases)==384
        for case in cases:
            game=from_spec(case['reference']);result=game.solve(first=case['first'])
            assert json.loads(json.dumps(result))==case['result']
        assert frozen_hashes()==original
    paths=[ROOT/'experiments/core/mechanism'/name for name in ('signaling_game.py','token_signaling_adapter.py')]
    paths += [ROOT/'experiments/mechanism'/name for name in ('audit_signaling_game.py','probe_signaling_interface.py',
        'probe_signaling_interface_v2.py','probe_signaling_interface_v3.py','replay_signaling_probe.py','complete_signaling_foundation.py')]
    paths += [ROOT/'experiments/tests'/name for name in ('test_signaling_game.py','test_signaling_interface.py','test_token_signaling_adapter.py')]
    paths += [ROOT/'docs/research/语义共识博弈_P0P1规格与验收_2026-10-24.md']
    paths += probe_files+list(fixtures.glob('*.json'))+list((ROOT/'experiments/results/audit/signaling_probe_replay_v1').glob('*.json'))
    OUT.mkdir(parents=True,exist_ok=True)
    result={'status':'P0_P1_baseline_implemented_partial_interface_not_efficacy_validated',
            'tests_passed':run.testsRun,'test_log':log.getvalue(),'original_frozen_files_verified':len(original),
            'prior_snapshots_verified':len(prior),'recent_acceptances_verified':recent,
            'http_attempts':attempts,'http_responses':responses,'known_token_lower_bound':known,
            'unknown_usage_responses':unknown,'unresolved_attempts':unresolved,
            'new_benchmark_calls':0,'new_calls_during_acceptance':0,
            'sha256':{str(p.relative_to(ROOT)):fingerprint(p) for p in paths}}
    freeze(OUT/'acceptance.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('sha256','test_log','recent_acceptances_verified')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
