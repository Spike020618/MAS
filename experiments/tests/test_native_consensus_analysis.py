import unittest
from experiments.paired_deliberation.native_consensus_analysis import cluster_difference
class NativeClusterAnalysisTests(unittest.TestCase):
    def test_multiple_prefixes_of_same_dialogue_remain_one_cluster(self):
        labels=[{'id':'conv:1','targets':[5,5]},{'id':'conv:9','targets':[6]}]
        r=cluster_difference(labels,[[5],[6]],[[],[]],replicates=20)
        self.assertEqual(r['dialogue_clusters'],1);self.assertEqual(r['item_targets'],3)
        self.assertEqual(r['recall@10_delta'],1);self.assertEqual(r['recall@10_cluster95_interval'],[1,1])
    def test_identical_methods_have_exactly_zero_paired_difference(self):
        labels=[{'id':'a:1','targets':[5]},{'id':'b:9','targets':[6,7]}]
        r=cluster_difference(labels,[[5],[6]],[[5],[6]],replicates=100)
        self.assertEqual(r['recall@10_delta'],0);self.assertEqual(r['ndcg@10_cluster95_interval'],[0,0])
if __name__=='__main__':unittest.main()
