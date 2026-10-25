from fractions import Fraction
import unittest
from experiments.mechanism.audit_incentive_necessity import (
    simplex,score,expected_score,projection,best_response,sandbagging_fixture,
)


class IncentiveNecessityTest(unittest.TestCase):
    def test_exact_simplex_no_normalization(self):
        self.assertTrue(simplex([.8,.2]))
        self.assertFalse(simplex([.9999,0.]))
        self.assertFalse(simplex([.8,.3]))

    def test_proper_score_identity(self):
        q=[.2,.3,.5];r=[.8,.1,.1]
        gap=expected_score(q,q)-expected_score(q,r)
        self.assertAlmostEqual(gap,.5*sum((a-b)**2 for a,b in zip(q,r)))
        self.assertTrue(0<=score(q,0)<=1)

    def test_absolute_score_keeps_truth(self):
        qs=[[.7,.3],[.6,.4],[.9,.1]]
        br=best_response('individual_absolute',qs[0],qs,['yes']*3,0,['yes','no'])
        self.assertEqual(br['strategy'],'honest')
        self.assertEqual(br['expected_component_gain'],0)

    def test_team_score_can_incentivize_exaggeration(self):
        qs=[[.7,.3],[.5,.5],[.5,.5]]
        br=best_response('team_brier_difference',qs[0],qs,['yes']*3,0,['yes','no'])
        self.assertGreater(br['expected_component_gain'],0)
        self.assertAlmostEqual(br['report'][0],1.)

    def test_majority_reward_can_incentivize_copy(self):
        qs=[[.4,.3,.3],[.1,.8,.1],[.1,.1,.8]]
        br=best_response('team_accuracy_difference',qs[0],qs,['a','b','c'],0,['a','b','c'])
        self.assertGreater(br['expected_component_gain'],0)
        self.assertIn(br['C'],['b','c'])

    def test_projection_and_budget_bound(self):
        r=projection([2.,-.2,.3])
        self.assertAlmostEqual(sum(r),1.)
        self.assertTrue(all(v>=0 for v in r))
        # S in [0,1] implies shifted difference payments in [0,B/n].
        for full in [0.,.2,1.]:
            for reference in [0.,.8,1.]:
                payment=Fraction(1,6)*(1+Fraction(str(full))-Fraction(str(reference)))
                self.assertTrue(0<=payment<=Fraction(1,3))

    def test_no_gold_in_best_response_interface(self):
        with self.assertRaises(TypeError):best_response('individual_absolute',[.5,.5],[[.5,.5]]*3,['yes']*3,0,['yes','no'],label='yes')

    def test_delta_payment_sandbagging(self):
        fixture=sandbagging_fixture()
        self.assertGreater(fixture['profit_from_lowering_initial_report'],0)

    def test_weighted_team_bonus_not_automatically_truthful(self):
        # Individual score has quadratic loss near q; team score can have a
        # nonzero first-order gain. Large own-score weight is no general proof.
        q=[.7,.3];others=[[.5,.5],[.5,.5]];eps=1e-6;r=[q[0]+eps,q[1]-eps]
        before=[(q[j]+sum(v[j] for v in others))/3 for j in range(2)]
        after=[(r[j]+sum(v[j] for v in others))/3 for j in range(2)]
        gain=100*(expected_score(q,r)-expected_score(q,q))+.01*(expected_score(q,after)-expected_score(q,before))
        self.assertGreater(gain,0)
