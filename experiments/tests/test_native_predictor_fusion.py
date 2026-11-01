import json,tempfile,unittest
from pathlib import Path
import numpy as np
from experiments.paired_deliberation.native_predictor_fusion import read_array_prefix,rrf_order,complementary,score_card

class PredictorFusionTests(unittest.TestCase):
    def test_selection_prefix_does_not_parse_later_evaluation_objects(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'labels.json';p.write_text('[{"id":"dev:1","targets":[7],"nested":{"brace":"}"}}, THIS LATER PART IS NOT VALID JSON]')
            self.assertEqual(read_array_prefix(p,1)[0]['targets'],[7])
            with self.assertRaises(ValueError):read_array_prefix(p,2)
    def test_weight_endpoints_reproduce_each_full_universe_order(self):
        sparse=np.array([[1,2,3,4]]);kbrd=np.array([[4,3,2,1]])
        self.assertEqual(rrf_order(sparse,kbrd,1.,60).tolist(),[[0,1,2,3]])
        self.assertEqual(rrf_order(sparse,kbrd,0.,60).tolist(),[[3,2,1,0]])
        with self.assertRaises(ValueError):rrf_order(sparse,kbrd,2.,60)
    def test_complementarity_retains_duplicate_target_denominator(self):
        labels=[{'id':'a','targets':[1,1,2,3,4]}]
        s=complementary(labels,[[1,2]],[[1,3]])
        self.assertEqual(s['item_targets'],5)
        self.assertEqual(s['top10']['both'],2);self.assertEqual(s['top10']['sparse_only'],1)
        self.assertEqual(s['top10']['kbrd_only'],1);self.assertEqual(s['top10']['neither'],1)
    def test_current_query_score_card_never_claims_current_user_truth(self):
        c=score_card('kbrd','dev:1',['7'],['movie_id','score'],[['7',.9]],{'fit_split':'train'})
        self.assertEqual(c['query_case_id'],'dev:1');self.assertNotEqual(c['origin_case_id'],'dev:1')
        self.assertFalse(c['current_user_fact_authority']);self.assertFalse(c['semantic_truth_certified'])
if __name__=='__main__':unittest.main()
