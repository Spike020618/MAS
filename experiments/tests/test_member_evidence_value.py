import unittest
import numpy as np
from experiments import member_evidence_value as v,member_evidence_data as d
from experiments.consensus_mechanism import core as c
class SpyChannel:
    def __init__(self):self.calls=[]
    def probs(self,X):
        rows=[]
        for f in X:
            self.calls.append(f)
            k=next(i for i,a in enumerate(v.ANSWERS)if f.get('hypothesized_world:'+a))
            rows.append({k:1.})
        return rows
class Validity:
    def probs(self,X):return [{1:.8,0:.2}for _ in X]
class ChannelTests(unittest.TestCase):
    def test_possible_worlds_enumerated_without_actual_truth(self):
        task=d.read(d.DEFAULT/'tasks.json')[0];state=c.fresh(task);c.expose(task,state,task['metadata']['seed_ids'])
        reports=[{'decision':'not_enough_info','claim':'Unknown','span_ids':[]}for _ in range(3)]
        spy=SpyChannel();q=v.response_channel({'review':spy,'validity':Validity()},task,state,reports,0,task['metadata']['acquisition_ids'][:1])
        self.assertEqual(q.shape,(3,4));self.assertTrue(np.allclose(q.sum(axis=1),1))
        self.assertTrue(np.allclose(q[:,3],.2));self.assertEqual(len(spy.calls),3)
        self.assertFalse(any('gold'in k for f in spy.calls for k in f))
    def test_invalid_outcome_does_not_update_belief(self):
        p=np.array([.2,.5,.3]);q=np.ones((3,4))*.25
        self.assertTrue(np.array_equal(v.update(p,q,{'valid':False}),p))
    def test_valid_evidence_updates_by_same_frozen_channel(self):
        p=np.array([.4,.3,.3]);q=np.array([[.8,.1,.1,0],[.1,.8,.1,0],[.1,.1,.8,0]])
        result=v.update(p,q,{'valid':True,'raw_report':{'decision':'refuted'}})
        expected=p*q[:,1];expected/=expected.sum()
        self.assertTrue(np.allclose(result,expected));self.assertEqual(int(np.argmax(result)),1)
if __name__=='__main__':unittest.main()
