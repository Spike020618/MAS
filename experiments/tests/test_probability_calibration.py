import unittest
from experiments.calibrated_action_value import OutcomeModel
from experiments.action_value_probability_calibration import SignedRiskOutcome,calibrate

class CalibrationTests(unittest.TestCase):
    def base(self):
        return {'outcome':OutcomeModel().fit([{'x':1.}]*3,[-1,0,1]),'validity':OutcomeModel().fit([{}],[1]),'cost_weight':.1}
    def test_probabilities_normalized_and_bounded(self):
        model=SignedRiskOutcome(self.base()['outcome'],.6)
        p=model.probs([{}])[0]
        self.assertAlmostEqual(sum(p.values()),1.);self.assertTrue(all(0<=x<=1 for x in p.values()))
        self.assertGreater(p[1],p[-1])
    def test_zero_only_training_not_forced_positive(self):
        model=SignedRiskOutcome(OutcomeModel().fit([{}],[0]),2.)
        self.assertEqual(model.probs([{}])[0],{-1:0.,0:1.,1:0.})
    def test_calibration_uses_only_valid_calibration_rows(self):
        rows=[{'partition':'calibration','valid':True,'task_id':'c','features':{},'reward':1}]
        model,stats=calibrate(self.base(),rows)
        other=rows+[{'partition':'test','valid':True,'task_id':'x','features':{},'reward':-1},
                    {'partition':'calibration','valid':False,'task_id':'bad','features':{},'reward':None}]
        _,second=calibrate(self.base(),other)
        self.assertEqual(stats,second);self.assertGreater(stats['signed_log_odds_bias'],0)
        self.assertLess(stats['loss_after'],stats['loss_before'])
    def test_claim_weighting_not_duplicate_count_weighting(self):
        a={'partition':'calibration','valid':True,'task_id':'a','features':{},'reward':1}
        b={**a,'task_id':'b','reward':-1}
        _,first=calibrate(self.base(),[a,b]);_,second=calibrate(self.base(),[a]*10+[b])
        self.assertAlmostEqual(first['signed_log_odds_bias'],second['signed_log_odds_bias'],places=5)
if __name__=='__main__':unittest.main()
