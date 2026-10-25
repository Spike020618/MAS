import math
import random
import unittest

from experiments.core.mechanism.signaling_game import SignalingGame
from experiments.core.mechanism.signaling_safety_audit import rank_lock,quarantine


class SignalingSafetyTests(unittest.TestCase):
    def test_exact_response_rank_lock_for_all_binary_checker_endpoints(self):
        game=SignalingGame(('a','b'),((.95,.05),(.1,.9)),((.5,.5),(.5,.5)))
        self.assertTrue(rank_lock(game.candidates,game.generator_reference[0],.5)['exact_response_rank_locked'])
        for i in range(1,100):
            for j in (1,50,99):
                response=game.best_generator(((i/100,1-i/100),(j/100,1-j/100)))
                self.assertGreater(response[0][0],response[0][1])

    def test_unlocked_does_not_claim_ability_to_correct(self):
        self.assertFalse(rank_lock(('a','b'),(.6,.4),.5)['exact_response_rank_locked'])
        self.assertAlmostEqual(rank_lock(('a','b'),(.6,.4),.5)['maximum_opponent_log_odds_shift'],2)

    def test_missing_input_cannot_be_called_a_game_success(self):
        r=quarantine('a','b',inputs_complete=False,converged=False)
        self.assertIsNone(r['reported_game_answer'])
        self.assertEqual(r['operational_fallback_answer'],'b')
        self.assertIn('not an effective',r['contribution'])

    def test_disagreement_not_silently_relabelled_as_verified(self):
        r=quarantine('a','b',inputs_complete=True,converged=True)
        self.assertFalse(r['game_admissible_as_unchanged_checker_readout'])
        self.assertEqual(r['reason'],'unresolved_independent_checker_disagreement')

    def test_fallback_equivalence_is_not_a_new_algorithm(self):
        rng=random.Random(20261024)
        for _ in range(1000):
            a,b=rng.choice(('a','b','abstain')),rng.choice(('a','b','abstain'))
            result=quarantine(a,b,inputs_complete=rng.choice((True,False)),converged=rng.choice((True,False)))
            self.assertEqual(result['operational_fallback_answer'],b)

    def test_invalid_reference_rejected(self):
        for strength in (0,-1,math.nan):
            with self.assertRaises(ValueError):rank_lock(('a','b'),(.5,.5),strength)

    def test_registered_candidate_order_not_json_dictionary_order(self):
        import json
        scores=json.loads(json.dumps({'yes':.95,'no':.05},sort_keys=True))
        self.assertEqual(list(scores),['no','yes'])
        result=rank_lock(('yes','no'),(.95,.05),.5)
        self.assertEqual(result['anchor_top'],'yes')

    def test_uniform_anchor_repair_collapses_to_direct_checker(self):
        from experiments.core.mechanism.signaling_repair_controls import frozen_checker_controls
        rng=random.Random(20261024)
        for n in range(2,11):
            for _ in range(20):
                d=[]
                for i in range(n):
                    p=.05+.9*rng.random();d.append((p,1-p))
                out=frozen_checker_controls(tuple(map(str,range(n))),((1/n,)*n,)*2,d)
                self.assertTrue(out['uniform_readout_equals_checker'])

    def test_freezing_checker_does_not_remove_anchor_obstruction(self):
        from experiments.core.mechanism.signaling_repair_controls import frozen_checker_controls
        out=frozen_checker_controls(('wrong','right'),((.99,.01),(.01,.99)),((.1,.9),(.9,.1)))
        self.assertEqual(out['answers']['freeze_checker'],'wrong')
        self.assertEqual(out['answers']['uniform_anchor_freeze_checker'],'right')
        self.assertEqual(out['direct_checker'],'right')


if __name__=='__main__':unittest.main()
