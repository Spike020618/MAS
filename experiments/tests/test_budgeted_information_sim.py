from dataclasses import asdict
import inspect
import unittest

from experiments.core.mechanism.budgeted_information_sim import (
    Contract, Decision, Episode, Estimate, Observation, ReplayResource, Report,
    choose, valid_probability,
)


def fixture():
    observations = [Observation('x', i, ('a', 'b'), Report('a', (.6, .4)), 100) for i in range(3)]
    broker = ReplayResource({('x', i): (Report('b', (.1, .9)), 200, 'sealed-source') for i in range(3)})
    return observations, broker


class BudgetedInformationTest(unittest.TestCase):
    def test_no_gold_or_other_agent_in_decision_interface(self):
        self.assertEqual(set(inspect.signature(choose).parameters), {'observation', 'estimate', 'contract', 'conservative'})
        self.assertNotIn('label', asdict(fixture()[0][0]))
        self.assertEqual(set(asdict(fixture()[0][0])), {'case_id', 'agent', 'choices', 'initial', 'initial_tokens'})

    def test_action_from_own_estimated_utility(self):
        o = fixture()[0][0]
        estimate = Estimate(.2, 200, 10, -.1, 'fixture')
        self.assertEqual(choose(o, estimate, Contract(1000)).action, 'K')
        self.assertEqual(choose(o, estimate, Contract(1001)).action, 'U')
        self.assertEqual(choose(o, estimate, Contract(10000), True).action, 'K')

    def test_no_incentive_means_no_upgrade(self):
        self.assertEqual(choose(fixture()[0][0], Estimate(.8, 1, 20, .2, 'fixture'), Contract(0)).action, 'K')

    def test_private_budget_blocks_profitable_action(self):
        o = fixture()[0][0]
        d = choose(o, Estimate(.8, 200, 20, .2, 'fixture'), Contract(10000, 2099, 2000))
        self.assertEqual(d.action, 'K')
        self.assertEqual(d.reason, 'insufficient_private_budget')

    def test_no_probability_normalization(self):
        self.assertFalse(valid_probability((.9999, 0.), 2))
        o = Observation('x', 0, ('a', 'b'), Report('a', (.9999, 0.)), 100)
        self.assertEqual(choose(o, Estimate(1., 1, 20, 1., 'fixture'), Contract(10000)).action, 'K')
        self.assertEqual(o.initial.score('a', o.choices), 0.)

    def test_no_upgrade_released_on_keep(self):
        observations, broker = fixture()
        e = Episode(observations, Contract(1000))
        e.submit(0, Decision('K', 0, 'test'), broker)
        # Still purchasable by trusted broker: no earlier receipt existed.
        _, receipt = broker.purchase('x', 0, 2000)
        self.assertEqual(receipt['tokens'], 200)

    def test_duplicate_purchase_and_overspend_fail_closed(self):
        _, broker = fixture()
        with self.assertRaises(ValueError): broker.purchase('x', 0, 199)
        broker.purchase('x', 0, 200)
        with self.assertRaises(ValueError): broker.purchase('x', 0, 200)

    def test_no_early_aggregate_or_settle(self):
        observations, broker = fixture()
        e = Episode(observations, Contract(1000))
        with self.assertRaises(ValueError): e.aggregate()
        with self.assertRaises(ValueError): e.settle('a')
        for i in range(3): e.submit(i, Decision('K', 0, 'test'), broker)
        with self.assertRaises(ValueError): e.settle('a')
        self.assertEqual(e.aggregate(), 'a')
        with self.assertRaises(ValueError): e.aggregate()

    def test_accounting_sealed_then_delayed(self):
        observations, broker = fixture()
        e = Episode(observations, Contract(1000))
        for i in range(3): e.submit(i, Decision('U' if i == 0 else 'K', 0, 'test'), broker)
        with self.assertRaises(ValueError): e.submit(0, Decision('K', 0, 'test'), broker)
        self.assertEqual(e.aggregate(), 'a')
        result = e.settle('b')
        self.assertEqual([r['tokens'] for r in result['records']], [300, 100, 100])
        self.assertEqual([r['wallet_remaining'] for r in result['records']], [2700, 2900, 2900])
        self.assertEqual([x['event'] for x in result['events']], ['sealed']*3+['aggregate', 'delayed_settlement'])
        self.assertAlmostEqual(result['records'][0]['utility'], 1000*.99-300)
        self.assertLessEqual(result['total_credit'], result['max_credit'])
        with self.assertRaises(ValueError): e.settle('b')

    def test_flipping_gold_changes_settlement_not_action(self):
        outputs = []
        for y in ('a', 'b'):
            observations, broker = fixture()
            contract = Contract(3000)
            ds = [choose(o, Estimate(.2, 200, 10, -.1, 'fixture'), contract) for o in observations]
            e = Episode(observations, contract)
            commitments = [e.submit(o.agent, d, broker) for o, d in zip(observations, ds)]
            prediction = e.aggregate()
            outputs.append((ds, commitments, prediction, e.settle(y)))
        self.assertEqual(outputs[0][:3], outputs[1][:3])
        self.assertNotEqual(outputs[0][3]['total_credit'], outputs[1][3]['total_credit'])

    def test_score_proper_under_fixed_subjective_belief(self):
        q = (.8, .2)
        for r in ((.5, .5), (1., 0.), (.1, .9)):
            truthful = sum(q[i]*Report('a', q).score(c, ('a', 'b')) for i, c in enumerate(('a', 'b')))
            answer = 'a' if r[0] >= r[1] else 'b'
            alternative = sum(q[i]*Report(answer, r).score(c, ('a', 'b')) for i, c in enumerate(('a', 'b')))
            self.assertAlmostEqual(truthful-alternative, .5*sum((a-b)**2 for a, b in zip(q, r)))

    def test_wallets_cannot_cross_subsidize(self):
        observations, broker = fixture()
        observations[0] = Observation('x', 0, ('a', 'b'), Report('a', (.6, .4)), 1500)
        e = Episode(observations, Contract(1000))
        with self.assertRaises(ValueError): e.submit(0, Decision('U', 0, 'test'), broker)
        e.submit(1, Decision('U', 0, 'test'), broker)

    def test_invalid_contract_rejected(self):
        for weight in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError): Contract(weight)

    def test_cannot_keep_paid_probability_but_flip_conclusion(self):
        observations, broker = fixture()
        observations[0] = Observation('x', 0, ('a', 'b'), Report('b', (.9, .1)), 100)
        self.assertEqual(observations[0].initial.score('a', ('a', 'b')), 0.)
        e = Episode(observations, Contract(1000))
        for i in range(3): e.submit(i, Decision('K', 0, 'test'), broker)
        e.aggregate()
        records = e.settle('a')['records']
        self.assertEqual(records[0]['accepted_answer'], 'abstain')
        self.assertEqual(records[0]['tokens'], 100)
        self.assertTrue(records[0]['rejected_inconsistent_conclusion'])

    def test_invalid_return_is_still_charged_and_raw_report_preserved(self):
        observations, _ = fixture()
        broker = ReplayResource({('x', 0): (Report('b', (.9, .1)), 200, 'source-bad-C')})
        e = Episode(observations, Contract(1000))
        e.submit(0, Decision('U', 1, 'test'), broker)
        for i in (1, 2): e.submit(i, Decision('K', 0, 'test'), broker)
        e.aggregate()
        r = e.settle('a')['records'][0]
        self.assertEqual(r['tokens'], 300)
        self.assertEqual(r['report']['answer'], 'b')
        self.assertEqual(r['accepted_answer'], 'abstain')
        self.assertEqual(r['score'], 0.)

    def test_no_model_or_network_dependency(self):
        from experiments.core.mechanism import budgeted_information_sim as module
        source = inspect.getsource(module)
        self.assertNotIn('import requests', source)
        self.assertNotIn('experiments.core.llm', source)


if __name__ == '__main__': unittest.main()
