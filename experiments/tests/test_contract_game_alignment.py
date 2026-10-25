import unittest
from itertools import product

from experiments.core.mechanism.contracts import CONTRACTS, scale_rewards
from experiments.core.mechanism.controller import MechanismController
from experiments.core.mechanism.schema import AEICReport, AgentAction
from experiments.mechanism.audit_contract_game_alignment import static_bounds


class ContractAlignmentTests(unittest.TestCase):
    def offer(self, scale):
        spec = scale_rewards(CONTRACTS['static_aeic'], scale)
        return {'credit_rule': spec.credit_rule, 'visible_reward': spec.rewards,
                'visible_cost': spec.costs}

    def test_frozen_scale_dominates_high_resource(self):
        bounds = static_bounds(self.offer(.2))
        self.assertAlmostEqual(bounds['aeic_effort']['max_utility'], -.6)
        self.assertAlmostEqual(bounds['verify']['max_utility'], -1.6)
        self.assertTrue(bounds['aeic_effort']['strictly_dominated_by_low_for_any_report'])
        self.assertTrue(bounds['verify']['strictly_dominated_by_low_for_any_report'])
        self.assertFalse(bounds['defer_imitate']['strictly_dominated_by_low_for_any_report'])

    def test_do_not_extend_bounds_to_other_contracts(self):
        with self.assertRaises(ValueError): static_bounds({'credit_rule': 'final_outcome'})

    def test_no_dominance_claim_for_large_reward(self):
        self.assertFalse(static_bounds(self.offer(2))['aeic_effort']['strictly_dominated_by_low_for_any_report'])

    def test_separable_payoffs_have_only_trivial_potential(self):
        utilities = [[0, -.6], [.1, -.7], [0, -1.6]]
        for profile in product((0, 1), repeat=3):
            before = sum(u[a] for u, a in zip(utilities, profile))
            for i in range(3):
                changed = list(profile)
                changed[i] = 1 - profile[i]
                after = sum(u[a] for u, a in zip(utilities, changed))
                self.assertAlmostEqual(after-before,
                    utilities[i][changed[i]]-utilities[i][profile[i]])

    def test_static_settlement_independent_of_peers_and_answer_key(self):
        spec = scale_rewards(CONTRACTS['static_aeic'], .2)
        report = AEICReport('premise', ['evidence'], 'inference', 'Yes')
        results = []
        for truth, peers in [('yes', ['same']), ('no', ['opposite']), (None, [])]:
            controller = MechanismController(2)
            results.append(controller.settle(0, spec.to_offer('fixed'),
                AgentAction.AEIC_EFFORT, report, truth, peers))
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[1], results[2])
        self.assertAlmostEqual(results[0]['utility'], -.6)

    def test_consistency_signal_is_not_semantic_verification(self):
        spec = scale_rewards(CONTRACTS['static_aeic'], .2)
        report = AEICReport('a', ['unverified'], 'i', 'Yes')
        result = MechanismController(1).settle(0, spec.to_offer('fixed'),
            AgentAction.LOW_EFFORT, report, 'no')
        self.assertEqual(result['component_consistency'], 1)
        self.assertEqual(result['credit'], .4)

    def test_static_contract_not_adapted_by_reputation(self):
        spec = scale_rewards(CONTRACTS['static_aeic'], .2)
        controller = MechanismController(2)
        controller.reputation[0] = -100
        controller.reputation[1] = 100
        self.assertEqual(controller.adaptive_spec(spec, 0), spec)
        self.assertEqual(controller.adaptive_spec(spec, 1), spec)


if __name__ == '__main__': unittest.main()
