import hashlib
import unittest
from experiments.core.mechanism.source_sharing import AcquisitionLedger, SourceSharingGame


class SourceSharingRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.data = b'fixture source, not a factual QA experiment'
        self.source = hashlib.sha256(self.data).hexdigest()
        self.game = SourceSharingGame(['a', 'b', 'c'],
            {a: [self.source] for a in ['a', 'b', 'c']}, {self.source: 1},
            {a: [0, .75] for a in ['a', 'b', 'c']}, 1)

    def test_local_execution_and_real_byte_receipts(self):
        run = self.game.run_local_updates()
        self.assertEqual(run['terminal']['profile'], [1, 0, 0])
        self.assertEqual(run['regrets'], [0, 0, 0])
        ledger = AcquisitionLedger('fixture', self.game, {self.source: self.data})
        self.assertEqual(ledger.acquire('a', self.source), self.data)
        ledger.acquire('a', self.source)
        result = ledger.settle([1, 0, 0])
        self.assertEqual(result['paid'], 1)
        self.assertEqual(len(result['receipts']), 1)

    def test_no_forged_or_missing_acquisition(self):
        ledger = AcquisitionLedger('fixture', self.game, {self.source: self.data})
        with self.assertRaises(ValueError): ledger.settle([1, 0, 0])
        with self.assertRaises(PermissionError): ledger.acquire('outsider', self.source)
        with self.assertRaises(ValueError): AcquisitionLedger('fixture', self.game, {self.source: b'changed'})

    def test_all_initial_profiles_and_orders(self):
        from itertools import permutations, product
        for initial in product((0, 1), repeat=3):
            for order in permutations(self.game.agents):
                run = self.game.run_local_updates(initial, order)
                self.assertEqual(max(run['regrets']), 0)

    def test_metadata_validation(self):
        with self.assertRaises(ValueError): self.game.run_local_updates(order=['a', 'a', 'b'])
        with self.assertRaises(ValueError): self.game.evaluate([1, 0])

    def test_cross_check_independent_nash_enumerator(self):
        from itertools import product
        from experiments.mechanism.solve_multifollower import solve_multifollower
        rows = [dict(self.game.evaluate(p), leader=0) for p in product((0, 1), repeat=3)]
        result = solve_multifollower({'agents': list(self.game.agents),
            'actions': {a: [0, 1] for a in self.game.agents},
            'contracts': [{'id': 'fixture', 'payoffs': rows}]})
        independent = {tuple(row['profile']) for row in result['contracts'][0]['pure_nash']}
        self.assertEqual(independent, {tuple(row['profile']) for row in self.game.equilibria()})

    def test_prepared_pilot_does_not_put_gold_in_messages(self):
        from experiments.mechanism.pilot_source_sharing import prepare
        cases, jobs = prepare()
        self.assertEqual((len(cases), len(jobs)), (6, 72))
        import json
        for job in jobs:
            content = json.loads(job['messages'][1]['content'])
            self.assertEqual(set(content), {'question', 'sources'})
            self.assertNotIn('ground_truth', content)
            if job['action'] == 'D': self.assertEqual(content['sources'], [])

    def test_neutral_repair_is_uniform_and_does_not_change_sources(self):
        from experiments.mechanism.pilot_source_sharing import prepare
        from experiments.mechanism.pilot_source_sharing_neutral import prepare as neutral
        _, original = prepare()
        _, repaired = neutral()
        for a, b in zip(original, repaired):
            self.assertEqual(a['messages'][1], b['messages'][1])
            self.assertEqual(a['key'], b['key'])
            self.assertNotEqual(a['prompt_sha256'], b['prompt_sha256'])
            self.assertIn('absence of supplied sources does not itself require abstention', b['messages'][0]['content'])


if __name__ == '__main__': unittest.main()
