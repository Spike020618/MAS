import unittest,copy
from experiments.evidence_consensus.state import new_state
from experiments.evidence_consensus.actions import fetch_source
from experiments.evidence_consensus.grounded_update import grounded_patch

class GroundedUpdateTests(unittest.TestCase):
 def state(self):
  task={'id':'q','kind':'qa','prompt':'question','options':['yes','no'],'initial':'yes','source_catalog':{k:{'source_id':k,'text':k,'scope':{'task_id':'q'}}for k in ['s1','s2']}}
  s=new_state(task)
  for k in ['s1','s2']:fetch_source(s,k)
  return s
 def report(self,s,value,source):
  return {'E':[source],'I':[{'source_id':source,'relation':'support'}],'C':{'proposed_patch':{'operation':'set_answer','value':value,'expected_version':s['version'],'evidence_ids':[source]}}}
 def test_reversal_requires_new_cited_evidence(self):
  s=self.state();r=grounded_patch(s,self.report(s,'no','s1'),'s1');self.assertTrue(r['accepted']);s=r['state'];before=copy.deepcopy(s)
  bad=grounded_patch(s,self.report(s,'yes','s1'),'s1');self.assertFalse(bad['accepted']);self.assertEqual(bad['state'],before)
  good=grounded_patch(s,self.report(s,'yes','s2'),'s2');self.assertTrue(good['accepted']);self.assertFalse(good['semantic_truth_certified'])
 def test_fetch_without_use_is_rejected(self):
  s=self.state();r=grounded_patch(s,self.report(s,'no','s1'),'s2');self.assertFalse(r['accepted']);self.assertEqual(r['state'],s)
 def test_stale_version_remains_rejected(self):
  s=self.state();r=self.report(s,'no','s1');r['C']['proposed_patch']['expected_version']=8;self.assertFalse(grounded_patch(s,r)['accepted'])
