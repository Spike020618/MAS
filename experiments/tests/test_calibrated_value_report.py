import unittest
from experiments.calibrated_value_report import paired
class PairedTests(unittest.TestCase):
    def test_equal_predictions_no_effect(self):
        x=paired([1,0,1,0],[1,0,1,0])
        self.assertEqual(x['difference'],0.);self.assertEqual(x['bootstrap_95'],[0.,0.])
        self.assertEqual(x['exact_paired_sign_p'],1.)
    def test_all_wins_exact_probability(self):
        x=paired([1]*6,[0]*6)
        self.assertEqual(x['wins'],6);self.assertEqual(x['losses'],0)
        self.assertEqual(x['bootstrap_95'],[1.,1.]);self.assertEqual(x['exact_paired_sign_p'],.03125)
    def test_symmetric_swaps_cancel(self):
        x=paired([1,0,1,0],[0,1,0,1])
        self.assertEqual(x['difference'],0.);self.assertEqual(x['wins'],2);self.assertEqual(x['losses'],2)
if __name__=='__main__':unittest.main()
