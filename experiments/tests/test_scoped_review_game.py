import copy
from itertools import product
import json
import unittest

from experiments.paired_deliberation.scoped_evidence import cards, request, validate
from experiments.paired_deliberation.review_allocation_game import ReviewGame


class ScopedEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.original = {'message:1': {'kind': 'dialogue', 'role': 'seeker', 'text': 'I want Comedy.'},
            'message:2': {'kind': 'dialogue', 'role': 'recommender', 'text': 'You like Horror.'},
            'history:train:1': {'kind': 'historical_prediction_basis', 'text': 'I like Horror.'},
            'movie:1': {'kind': 'catalog', 'movie_id': '1', 'text': 'Movie 1 genre Comedy.'}}
        self.cards = cards('dev:1', self.original)
        self.req = request('dev:1', self.cards, {'a': '1', 'b': '2'})
        self.answer = {'preferred': '1', 'current_expression_ids': ['message:1'],
                       'catalog_ids': [], 'prediction_basis_ids': ['history:train:1']}

    def test_other_user_and_recommender_cannot_enter_current_expression_channel(self):
        for sid in ('history:train:1', 'message:2'):
            bad = copy.deepcopy(self.answer); bad['current_expression_ids'] = [sid]; bad['prediction_basis_ids'] = []
            with self.assertRaisesRegex(ValueError, 'scope/authority'): validate(bad, self.req)

    def test_origin_and_original_text_are_bound_by_runtime_not_generated(self):
        bound = validate(self.answer, self.req)
        self.assertEqual(bound['bound_sources'][1]['origin_case_id'], 'train:1')
        self.assertEqual(bound['bound_sources'][1]['text'], 'I like Horror.')
        self.assertFalse(bound['current_preference_certified'])
        self.assertFalse(bound['hard_constraint_created'])
        for field in ('reason', 'quote', 'current_user_claim'):
            with self.assertRaises(ValueError): validate({**self.answer, field: 'The current user likes Horror.'}, self.req)
        self.assertEqual(self.original['history:train:1']['text'], 'I like Horror.')


class ReviewGameTests(unittest.TestCase):
    def game(self):
        return ReviewGame([{'owner': 0, 'value': .9}, {'owner': 1, 'value': .7}, {'owner': 2, 'value': .5}], [[.1]*3]*3)

    def test_independent_exhaustive_potential_identity(self):
        g = self.game()
        for profile in product(*g.actions):
            for i in range(3):
                for a in g.actions[i]:
                    alt = profile[:i] + (a,) + profile[i+1:]
                    self.assertAlmostEqual(g.utility(i, alt)-g.utility(i, profile), g.potential(alt)-g.potential(profile))

    def test_actual_strategy_deviations_and_unprofitable_duplicate(self):
        g = self.game(); solved = g.solve(); a = solved['actions']
        self.assertTrue(solved['trace']); self.assertTrue(solved['finite_game_epsilon_nash'])
        self.assertLessEqual(max(g.deviations(a)), 1e-10)
        self.assertEqual(len([q for q in a if q is not None]), len({q for q in a if q is not None}))
        duplicate = (1, 0, 0)
        self.assertLess(g.utility(1, duplicate), 0)
        self.assertGreaterEqual(solved['central_potential_gap'], -1e-10)

    def test_high_cost_causes_real_abstention(self):
        g = ReviewGame([{'owner': 0, 'value': .1}], [[.2], [.2], [.2]])
        self.assertEqual(g.solve()['actions'], [None, None, None])


if __name__ == '__main__': unittest.main()
