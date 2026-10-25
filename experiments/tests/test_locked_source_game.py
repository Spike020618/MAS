import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.mechanism.validate_locked_source_game import canonical, freeze, run_case, valid_report


class FakeCaller:
    def __init__(self): self.calls = {}

    def call(self, key, messages):
        content = json.loads(messages[1]['content'])
        assert set(content) == {'question', 'sources'}
        row = {'key': key, 'parsed': {'C': 'yes', 'A': '', 'E': '', 'I': ''},
               'usage': {'total_tokens': 10}, 'wall_seconds': .01, 'utc': '2026-10-19'}
        self.calls[key] = row
        return row


class LockedSourceGameTests(unittest.TestCase):
    def test_online_chain_commitment_receipts_and_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            for name in ['snapshots', 'runs', 'commitments']: (directory/name).mkdir()
            data = b'<html>untrusted real bytes fixture</html>'
            source = hashlib.sha256(data).hexdigest()
            (directory/'snapshots'/f'{source}.html').write_bytes(data)
            bank = {'sources': [{'source_hash': source, 'title': 'fixture', 'url': 'https://example.org',
                                  'text': 'fixture', 'excerpt_sha256': hashlib.sha256(b'fixture').hexdigest()}]}
            protocol = {'agents': ['0', '1', '2'], 'costs': {str(i): [0, .75] for i in range(3)},
                        'initial': [0, 0, 0], 'order': ['0', '1', '2'], 'report_prompts': ['same']*3}
            case = {'question_id': 'fixture/one', 'question': 'A question?'}
            caller = FakeCaller()
            result = run_case(directory, protocol, case, bank, caller, 1)
            self.assertTrue(result['equilibrium_check_passed'])
            self.assertEqual(len(result['settlement']['receipts']), 1)
            self.assertEqual(len(caller.calls), 3)
            self.assertEqual(run_case(directory, protocol, case, bank, caller, 1), result)
            self.assertEqual(len(caller.calls), 3)
            changed = dict(protocol, order=['2', '1', '0'])
            with self.assertRaises(ValueError): run_case(directory, changed, case, bank, caller, 1)

    def test_frozen_config_cannot_change(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'frozen.json'
            freeze(path, {'budget': 1}); freeze(path, {'budget': 1})
            with self.assertRaises(ValueError): freeze(path, {'budget': 2})

    def test_schema_failure_and_canonical_matching(self):
        self.assertFalse(valid_report({'parsed': {'C': 'yes', 'A': [], 'E': '', 'I': ''}}))
        self.assertEqual(canonical('Is it TRUE?'), canonical('is IT true !'))

    def test_independent_payoff_arithmetic(self):
        from experiments.mechanism.audit_locked_source_game import arithmetic
        agents = ['0','1','2']
        bundles = {'0':['x'],'1':['y'],'2':['x','y']}
        costs = {a:[0,.75] for a in agents}
        credits, utilities = arithmetic([0,0,1],agents,bundles,['x','y'],costs,1)
        self.assertEqual(credits,[0,0,1])
        self.assertEqual(utilities,[0,0,.25])
        credits, utilities = arithmetic([1,0,1],agents,bundles,['x','y'],costs,1)
        self.assertEqual(credits,[.25,0,.75])
        self.assertEqual(utilities,[-.5,0,0])

    def test_development_optimum_reconstructed_from_raw_reports(self):
        from experiments.mechanism.audit_locked_source_game import verify_development
        root = Path(__file__).resolve().parents[2]
        path = root/'experiments/results/mechanism/stackelberg/locked_source_game_test_v1/protocol.json'
        result = verify_development(root,json.loads(path.read_text()))
        values = result['values']
        self.assertEqual(max(values,key=lambda r:r['pessimistic'])['budget'],1)


if __name__ == '__main__': unittest.main()
