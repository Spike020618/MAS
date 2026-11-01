import unittest
from unittest.mock import patch
from experiments.evidence_consensus import runner_general as r
class Predictor:
 def __init__(self,v):self.v=v
 def predict(self,x):return [self.v]
class GeneralRuntimeTests(unittest.TestCase):
 def test_learned_negative_value_abstains_but_fixed_unchanged(self):
  old=dict(r.ESTIMATOR)
  try:
   r.ESTIMATOR.update(mode='learned',model='unused')
   rows=[{'source_id':'s','score':1.,'source_characters':20}]
   task={'id':'q','kind':'qa','prompt':'q','source_catalog':{'s':{'text':'body'}},'candidates':[]}
   state={'decision':'yes','version':0,'visible_sources':{}}
   bundle={'feature_schema':'identity-free-structural-v1','gain':Predictor(-1),'damage':Predictor(0),'gain_bias':0}
   with patch('joblib.load',return_value=bundle):self.assertEqual(r.allocate_general(task,state,rows,'game')['actions'],[])
   self.assertEqual(r.allocate_general(task,state,rows,'fixed')['actions'],[(0,0)])
  finally:r.ESTIMATOR.clear();r.ESTIMATOR.update(old)
 def test_incompatible_schema_rejected(self):
  old=dict(r.ESTIMATOR)
  try:
   r.ESTIMATOR.update(mode='learned',model='unused')
   with patch('joblib.load',return_value={}):
    with self.assertRaisesRegex(ValueError,'schema'):r.allocate_general({}, {}, [],'central')
  finally:r.ESTIMATOR.clear();r.ESTIMATOR.update(old)
