import unittest
import numpy as np
from experiments.paired_deliberation.native_review_diagnostic_data import select_blind

class BlindDiagnosticTests(unittest.TestCase):
    def fixture(self):
        ids=['1:2','1:4','2:2','3:2'];pool=np.tile(np.arange(20),4);offset=np.arange(5)*20
        mean=np.tile(np.linspace(1,0,20),4);ens=np.stack([mean,mean+.02*np.sin(pool),mean-.02*np.sin(pool)])
        return ids,pool,offset,mean,ens
    def test_conversation_disjoint_and_label_free_deterministic(self):
        ids,pool,off,mean,ens=self.fixture()
        a=select_blind(ids,ids,pool,off,mean,ens,3);b=select_blind(ids,list(reversed(ids)),pool,off,mean,ens,3)
        self.assertEqual(a,b);self.assertEqual(len({p['case_id'].split(':')[0]for p in a}),3)
        for p in a:
            self.assertGreaterEqual(p['target']['a_position'],3);self.assertLess(p['target']['a_position'],10)
            self.assertGreaterEqual(p['target']['b_position'],10);self.assertNotIn('winner',p)
            self.assertFalse(p['query_labels_available'])
    def test_missing_query_rejected(self):
        ids,pool,off,mean,ens=self.fixture()
        with self.assertRaises(ValueError):select_blind(ids,['unknown'],pool,off,mean,ens,1)
    def test_nonfinite_forecast_rejected(self):
        ids,pool,off,mean,ens=self.fixture();mean[:]=float('nan')
        with self.assertRaises(ValueError):select_blind(ids,ids,pool,off,mean,ens,1)
if __name__=='__main__':unittest.main()
