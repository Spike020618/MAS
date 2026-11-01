import math,unittest
from experiments.paired_deliberation.expected_review_value import positive_part_normal,expected_review_value
class ExpectedReviewValueTests(unittest.TestCase):
    def test_independent_numerical_integral_of_noisy_comparison_gain(self):
        # Integrate actual signed gain conditional on the latent difference.
        for mu,sigma,q in [(-.3,.2,.7),(0.,.3,.5),(.2,.5,.8),(-.7,.4,1.)]:
            step=.002;observed=0.
            for i in range(-6000,6001):
                z=i*step;d=mu+sigma*z
                choose_b=q if d>0 else 1-q
                observed+=d*choose_b*math.exp(-z*z/2)/math.sqrt(2*math.pi)*step
            self.assertAlmostEqual(expected_review_value(mu,sigma,q,baseline='retain')['review_gain_over_retain'],observed,places=6)
    def test_no_uncertainty_review_cannot_beat_best_direct_decision(self):
        for mu in (-.2,0.,.2):
            for q in (.5,.8,1.):self.assertLessEqual(expected_review_value(mu,0.,q)['incremental_review_gain'],1e-12)
    def test_random_review_has_no_positive_value_for_a_better_incumbent(self):
        for mu in (-1.,-.1,0.):self.assertLessEqual(expected_review_value(mu,.5,.5)['net_value'],1e-12)
    def test_oracle_upper_is_not_reported_as_actual_gain(self):
        oracle=expected_review_value(0.,.4,1.);noisy=expected_review_value(0.,.4,.6)
        self.assertGreater(oracle['net_value'],noisy['net_value']);self.assertGreater(noisy['net_value'],0)
        self.assertEqual(oracle['assumed_symmetric_accuracy'],1.)
        self.assertFalse(expected_review_value(0.,.4,.6,cost=1)['participation_candidate'])
    def test_bad_parameters_rejected(self):
        for mu,s in [(float('nan'),1),(0,-1),(0,float('inf'))]:
            with self.assertRaises(ValueError):positive_part_normal(mu,s)
        for q in (-.1,1.1,float('nan')):
            with self.assertRaises(ValueError):expected_review_value(0,1,q)
if __name__=='__main__':unittest.main()

class AsymmetricReviewTests(unittest.TestCase):
    def test_symmetric_special_case_matches(self):
        from experiments.paired_deliberation.expected_review_value import asymmetric_review_value
        for mu,sigma,q in [(-.2,.4,.8),(.1,.2,.6),(0.,0.,1.)]:
            self.assertAlmostEqual(asymmetric_review_value(mu,sigma,q,q)['net_value'],expected_review_value(mu,sigma,q)['net_value'])
    def test_noop_reviewer_has_zero_gain_over_retention(self):
        from experiments.paired_deliberation.expected_review_value import asymmetric_review_value
        self.assertEqual(asymmetric_review_value(-.2,.5,0,1,baseline='retain')['net_value'],0.)
    def test_numeric_integral_without_symmetric_accuracy_assumption(self):
        from experiments.paired_deliberation.expected_review_value import asymmetric_review_value
        mu=-.1;sigma=.4;recover=.6;retain=.95;step=.002;observed=0.
        for i in range(-6000,6001):
            z=i*step;d=mu+sigma*z
            choose_b=recover if d>0 else 1-retain
            observed+=d*choose_b*math.exp(-z*z/2)/math.sqrt(2*math.pi)*step
        self.assertAlmostEqual(asymmetric_review_value(mu,sigma,recover,retain,baseline='retain')['net_value'],observed,places=6)
