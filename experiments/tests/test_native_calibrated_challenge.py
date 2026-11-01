import unittest
from collections import Counter
import numpy as np
from experiments.paired_deliberation.native_calibrated_challenge import split_indices,conv,candidate_features,make_table,FEATURES,metrics_for,calibrated_targets,allocate_calibrated,normalize_mass

class CalibratedCountTests(unittest.TestCase):
    def table(self):
        ids=['a:1','b:1'];items=list(range(20));rows=[{'id':c,'text':'Seeker: comedy','context_movies':[],'context_entities':[],'last_seeker_text':'comedy'} for c in ids]
        labels=[{'id':ids[0],'targets':[1,1,2]},{'id':ids[1],'targets':[3]}]
        arrays={'sparse_scores':np.tile(np.arange(20,0,-1,dtype=float),(2,1)),
            'sparse_ranks':np.tile(np.arange(1,21,dtype=np.uint16),(2,1)),
            'kbrd_logits':np.tile(np.arange(20,0,-1,dtype=float),(2,1)),
            'kbrd_ranks':np.tile(np.arange(1,21,dtype=np.uint16),(2,1))}
        return make_table(ids,items,rows,labels,arrays)
    def test_repeated_occurrence_count_and_micro_macro_are_distinct(self):
        t=self.table();self.assertEqual(t['X'].shape,(40,len(FEATURES)))
        self.assertEqual(float(t['count'][1]),2);self.assertAlmostEqual(float(t['mass'][1]),2/3,places=6)
        score=np.zeros(40);chosen={1,4,5,6,7,8,9,10,11,12}
        for i,m in enumerate(t['candidate_ids']):score[i]=float(int(m) in chosen)
        metric=metrics_for(t,[0,1],score,calibration=False)
        self.assertEqual(metric['metrics']['recall@10'],.5);self.assertAlmostEqual(metric['prefix_macro_recall@10'],1/3)
    def test_conversation_partitions_never_split_two_prefixes(self):
        ids=[str(i)+':'+str(j) for i in range(200) for j in (1,7)];s=split_indices(ids)
        self.assertFalse({conv(ids[i]) for i in s['fit']} & {conv(ids[i]) for i in s['diagnostic']})
        self.assertFalse({conv(ids[i]) for i in s['inner_fit']} & {conv(ids[i]) for i in s['select']})
    def test_unsigned_rank_difference_is_safe_and_features_use_no_labels(self):
        t=self.table();self.assertTrue(np.isfinite(t['X']).all())
        self.assertTrue(np.all(t['X'][:,FEATURES.index('log_rank_gap')]==0))
    def test_expected_count_above_one_is_valid_and_strong_incumbent_harm_is_visible(self):
        ranking=list(map(str,range(20)));c={m:.01 for m in ranking};c['0']=2;c['11']=1.5
        targets=calibrated_targets(ranking,c,max_targets=1)
        self.assertEqual(len(targets),1);self.assertEqual(targets[0]['b'],'11')
        self.assertAlmostEqual(targets[0]['ideal_swap_gain'],1.49)
        self.assertEqual(targets[0]['gain_count'],1.5);self.assertEqual(targets[0]['harm_count'],.01)
        self.assertGreaterEqual(ranking.index(targets[0]['a']),3)
    def test_no_automatic_challenge_when_all_outside_candidates_are_worse(self):
        ranking=list(map(str,range(20)));c={m:1/(int(m)+1) for m in ranking}
        self.assertEqual(calibrated_targets(ranking,c),[])
        self.assertEqual(calibrated_targets(ranking,c,max_targets=0),[])
    def test_same_utility_table_game_and_central_explicit_cost_units(self):
        targets=[{'a':'a','b':'b','owner':0,'value':.2}];costs=[[.01],[.01],[.01]]
        g=allocate_calibrated(targets,costs,'game',expected_hits_per_cny=2)
        c=allocate_calibrated(targets,costs,'central',expected_hits_per_cny=2)
        self.assertEqual(g['selected_potential'],c['selected_potential']);self.assertEqual(g['cost_units'],'expected native hit count')
    def test_macro_mass_normalizer_does_not_cap_expected_count_interface(self):
        p=normalize_mass([.8,.8,.3],[0,0,1]);self.assertTrue(np.allclose(p,[.5,.5,.3]))
        with self.assertRaises(ValueError):calibrated_targets(list(map(str,range(20))),{str(i):float('nan') for i in range(20)})
if __name__=='__main__':unittest.main()
