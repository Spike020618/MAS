import importlib.util,unittest
import numpy as np
HAS=bool(importlib.util.find_spec('torch'))
if HAS:
    from experiments.paired_deliberation.native_gain_calibration_data import fold,export_ranks,audit_open

@unittest.skipUnless(HAS,'isolated KBRD environment required for fold generator')
class NativeGainDataTests(unittest.TestCase):
    def test_fold_is_by_conversation_not_target_or_prefix(self):
        a={'id':'conversation:1','targets':[1]};b={'id':'conversation:99','targets':[999]}
        self.assertEqual(fold(a),fold(b))
    def test_native_rank_columns_and_popularity_ties(self):
        values=np.array([[1,1,0],[0,2,2]],dtype=np.float32)
        rank=export_ranks(values,[10,20,30],np.array([1,0,2]))
        self.assertEqual(rank.dtype,np.uint16);self.assertEqual(rank.tolist(),[[2,1,3],[3,1,2]])
    def test_runtime_hook_forbids_public_and_legacy_targets(self):
        for path in ('/project/data/valid_data.json','/project/data/test_data.json','/project/reserved_test.inputs.json'):
            with self.assertRaises(RuntimeError):audit_open('open',(path,'r',0))
        audit_open('open',('/project/experiments/data/train_data.json','r',0))
if __name__=='__main__':unittest.main()
