import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from experiments.mechanism import audit_information_readout_gap as audit
from experiments.mechanism.run_upgrade_pilot import freeze


def main():
    prior=audit.source.OUT/'acceptance.json'
    manifest=json.loads(prior.read_text())
    for rel,value in manifest['sha256'].items():assert audit.source.h(audit.source.ROOT/rel)==value,rel
    with (patch('socket.socket.connect',side_effect=AssertionError('offline')),
          patch('socket.getaddrinfo',side_effect=AssertionError('offline'))):
        with patch('builtins.print'):audit.main()
        log=io.StringIO()
        tests=unittest.TextTestRunner(stream=log).run(unittest.defaultTestLoader.loadTestsFromNames(
            ['experiments.tests.test_information_readout_gap','experiments.tests.test_learned_production_game',
             'experiments.tests.test_learned_production_runner']))
        assert tests.wasSuccessful(),log.getvalue()
    result=json.loads((audit.OUT/'result.json').read_text())
    assert result['oracle_available_correct']==11
    assert result['available_but_initial_vote_wrong']==1
    assert result['metrics']['train_selected_single']['correct']==10
    paths=[p for p in audit.OUT.iterdir() if p.is_file() and p.name!='acceptance.json']
    paths += [Path(__file__),Path(audit.__file__),audit.source.ROOT/'experiments/tests/test_information_readout_gap.py',
              audit.source.ROOT/'docs/research/信息增量与读出断点_核查_2026-10-25.md',prior]
    accepted={'status':'posthoc_readout_audit_complete_not_new_efficacy_evidence','tests':tests.testsRun,
              'new_api_calls':0,'sha256':{str(p.relative_to(audit.source.ROOT)):audit.source.h(p) for p in paths}}
    freeze(audit.OUT/'acceptance.json',accepted)
    print(json.dumps({k:v for k,v in accepted.items() if k!='sha256'},indent=2))


if __name__=='__main__':main()
