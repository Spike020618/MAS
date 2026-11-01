import unittest,tempfile,json
from pathlib import Path
from experiments.fit_transfer_contribution import components,prepare,features
class TransferTests(unittest.TestCase):
 def task(self,i,text,native=None):return {'id':str(i),'kind':'qa','metadata':{'native_id':native or str(i)},'source_catalog':{'s':{'text':text}}}
 def test_shared_content_and_original_group_are_transitive(self):
  ts=[self.task(1,'a','same'),self.task(2,'b','same'),self.task(3,'b'),self.task(4,'c')]
  self.assertEqual(sorted(map(len,components(ts))),[1,3])
 def test_large_connected_corpus_fails_instead_of_leaking(self):
  with tempfile.TemporaryDirectory()as d:
   r=Path(d);(r/'inputs.json').write_text(json.dumps([self.task(i,'shared')for i in range(9)]))
   with self.assertRaisesRegex(ValueError,'three'):prepare(r,r/'out')
 def test_features_do_not_expose_identity(self):
  t=self.task(1,'content');c={'before_decision':'yes','event':{'expected_state_version':0,'source_action':{'source_id':'s','before_visible_ids':[]}}}
  self.assertEqual(set(features(t,c)),{'source_chars','visible_count','version','candidate_count','baseline_abstain'})
