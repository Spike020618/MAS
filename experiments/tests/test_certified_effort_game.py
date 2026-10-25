from copy import deepcopy
from fractions import Fraction as F
from itertools import product
import unittest
import random

from experiments.core.mechanism.certified_effort_game import (
    PROFILES, empirical_game, equilibria, majority, pivotal_credits, regret,
    select_equilibrium, strict_reply_path,
    FollowerMenu, follower_reply, follower_menus, commit_recommendation,
)
from experiments.core.mechanism.budgeted_information_sim import Contract, Observation, Report
from experiments.mechanism.replay_budgeted_information import execute, forced_plan
from experiments.mechanism.replay_certified_effort_game import certify_potential, settle_traces


def cases():
    # Deliberate coordination complementarity: either high alone insufficient.
    return [{'id': 'x', 'choices': ['a', 'b'], 'low': ['a']*3, 'high': ['b']*3,
             'low_tokens': [1]*3, 'high_tokens': [2]*3}]


class CertifiedEffortGameTest(unittest.TestCase):
    def test_quorum_budget_over_all_registered_ballots_and_abstention(self):
        choices = ['a', 'b', 'c', 'd']
        for votes in product(choices+['abstain'], repeat=3):
            for y in choices:
                credits = pivotal_credits(list(votes), choices, y, F(7))
                self.assertTrue(all(0 <= v <= 7 for v in credits))
                self.assertLessEqual(sum(credits), 14)

    def test_exact_potential_action_edges_and_all_costs(self):
        table = empirical_game(cases(), {'x': 'b'}, F(5))
        self.assertEqual(certify_potential(table), 24)
        self.assertEqual(table[(1, 1, 0)]['utilities'], [F(2), F(2), F(-1)])
        self.assertEqual(table[(1, 1, 0)]['potential'], F(-2))

    def test_joint_success_can_coexist_with_all_keep_equilibrium(self):
        table = empirical_game(cases(), {'x': 'b'}, F(5))
        self.assertIn((0, 0, 0), equilibria(table))
        self.assertIn((1, 1, 0), equilibria(table))
        self.assertEqual(strict_reply_path(table), [(0, 0, 0)])
        self.assertEqual(select_equilibrium(table), (0, 1, 1))
        self.assertGreater(table[(0, 1, 1)]['potential'], table[(0, 0, 0)]['potential'])

    def test_true_action_NE_not_claimed_for_other_distribution(self):
        table = empirical_game(cases(), {'x': 'b'}, F(5))
        chosen = select_equilibrium(table)
        flipped = empirical_game(cases(), {'x': 'a'}, F(5))
        self.assertEqual(regret(table, chosen), 0)
        self.assertGreater(regret(flipped, chosen), 0)

    def test_individual_reward_is_separate_not_an_unstated_team_bonus(self):
        table = empirical_game(cases(), {'x': 'b'}, F(5))
        self.assertEqual(equilibria(table, 'individual_utilities'), [(1, 1, 1)])
        for row in table.values(): self.assertLessEqual(sum(row['individual_payments_same_cap']), 10)

    def test_no_reward_no_costly_purchase(self):
        table = empirical_game(cases(), {'x': 'b'}, F(0))
        self.assertEqual(equilibria(table), [(0, 0, 0)])

    def test_strict_replies_end_at_empirical_NE(self):
        table = empirical_game(cases(), {'x': 'b'}, F(5))
        for profile in PROFILES:
            path = strict_reply_path(table, profile)
            self.assertIn(path[-1], equilibria(table))
            self.assertEqual(len(path), len(set(path)))

    def test_generated_correlated_worlds_need_no_independent_signal_assumption(self):
        rng = random.Random(20261023)
        for _ in range(100):
            worlds, labels = [], {}
            for j in range(rng.randint(2, 8)):
                # Freely sampled joint path bundles, not a factored channel.
                choices = ['a', 'b', 'c']
                worlds.append({'id': str(j), 'choices': choices,
                    'low': rng.choices(choices+['abstain'], k=3),
                    'high': rng.choices(choices+['abstain'], k=3),
                    'low_tokens': [rng.randint(1, 10) for _ in range(3)],
                    'high_tokens': [rng.randint(1, 10) for _ in range(3)]})
                labels[str(j)] = rng.choice(choices)
            table = empirical_game(worlds, labels, F(rng.randint(0, 100)))
            self.assertEqual(certify_potential(table), 24)
            self.assertTrue(equilibria(table))
            for a in PROFILES:
                self.assertIn(strict_reply_path(table, a)[-1], equilibria(table))

    def test_unknown_or_forged_costs_do_not_become_free_resource_actions(self):
        for value in (None, 0, -1, 1.5):
            bad = deepcopy(cases())
            bad[0]['high_tokens'][0] = value
            with self.assertRaises(ValueError): empirical_game(bad, {'x': 'b'}, F(5))

    def test_selection_accepts_only_public_table_not_task_or_gold(self):
        import inspect
        self.assertEqual(tuple(inspect.signature(select_equilibrium).parameters), ('table',))
        self.assertEqual(tuple(inspect.signature(strict_reply_path).parameters), ('table', 'start', 'order'))

    def test_follower_decides_from_own_menu_not_a_forced_central_assignment(self):
        table = empirical_game(cases(), {'x': 'b'}, F(5))
        menus, responses = commit_recommendation(table, select_equilibrium(table))
        self.assertEqual(responses, select_equilibrium(table))
        self.assertEqual(len(menus), 3)
        with self.assertRaises(ValueError): commit_recommendation(table, (1, 1, 1))
        self.assertEqual(follower_reply(FollowerMenu(0, 1, F(3), F(2))), 0)
        self.assertEqual(follower_reply(FollowerMenu(0, 0, F(3), F(4))), 1)

    def test_private_budget_can_reject_otherwise_profitable_upgrade(self):
        menu = FollowerMenu(0, 1, F(0), F(100), 2999, 1000, 2000)
        self.assertEqual(follower_reply(menu), 0)

    def test_tool_report_edit_rejected_before_payment(self):
        obs = {'x': [Observation('x', i, ('a', 'b'), Report('a', (.7, .3)), 100) for i in range(3)]}
        high = {('x', i): (Report('b', (.1, .9)), 200, 'origin-'+str(i)) for i in range(3)}
        metadata = {('x', i, backend): {'seconds': 1} for i in range(3) for backend in ['zhipu_flash', 'deepseek']}
        profile = (1, 1, 0)
        plan = forced_plan(obs, {('x', 0), ('x', 1)})
        _, rows = execute(obs, high, plan, Contract(0), {'x': 'b'}, metadata)
        genuine = settle_traces(rows, profile, 1000, {'x': 'b'}, {'x': ('a', 'b')}, obs, high)
        self.assertEqual(genuine[0]['credits_total'], 2000)
        self.assertEqual([r['tokens'] for r in genuine[0]['ledgers']], [300, 300, 100])
        for field in ('report', 'tokens', 'receipt', 'accepted_answer', 'wallet_remaining'):
            changed = deepcopy(rows)
            record = changed[0]['settlement']['records'][0]
            if field == 'report': record['report']['answer'] = 'a'
            elif field == 'tokens': record['tokens'] = 0
            elif field == 'receipt': record['receipt']['source_response_hash'] = 'forged'
            elif field == 'accepted_answer': record['accepted_answer'] = 'a'
            else: record['wallet_remaining'] = 99999
            with self.assertRaises(ValueError):
                settle_traces(changed, profile, 1000, {'x': 'b'}, {'x': ('a', 'b')}, obs, high)


if __name__ == '__main__': unittest.main()
