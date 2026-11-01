import unittest
from experiments.paired_deliberation.native_reviewer_calibration import normalize,wilson
class ReviewerCalibrationTests(unittest.TestCase):
    def test_exact_representations_preserve_choice(self):
        obj,ops=normalize({'answer':{'preferred':123,'source_id':'e:1'}},['123','456'])
        self.assertEqual(obj,{'preferred':'123','source_id':'e:1'});self.assertEqual(len(ops),2)
        raw={'preferred':True,'source_id':'e:1'};self.assertEqual(normalize(raw,['1','2'])[0],raw)
        raw={'preferred':999,'source_id':'e:1'};self.assertEqual(normalize(raw,['123','456'])[0],raw)
    def test_unrelated_fields_and_sources_never_repaired(self):
        raw={'answer':{'preferred':'123','source_id':'missing','extra':'x'}}
        self.assertEqual(normalize(raw,['123','456']),(raw,[]))
    def test_conditional_rate_interval_is_not_certainty(self):
        self.assertLess(wilson(24,24)[0],1.);self.assertGreater(wilson(0,24)[1],0.)
        self.assertEqual(wilson(0,0),[0.,1.])
if __name__=='__main__':unittest.main()
