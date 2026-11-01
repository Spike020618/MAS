import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from experiments.evidence_consensus.runner_protocol_v2 import scope_proposals,allocate_general,ESTIMATOR,METHODS
from experiments.acquire_disjoint_actions import validate_inputs

class ProtocolV2Tests(unittest.TestCase):
 def test_unscoped_source_not_assigned_all_claims(self):
  with patch('experiments.evidence_consensus.runner_protocol_v2.propose_doubts',return_value=[{'source_ids':['s1'],'target_claim_ids':['c1'],'doubt_id':'d','question':'Does premise c1 hold?'}]):
   rows=scope_proposals({}, {'claims':{'c1':{},'c2':{}}},[],[{'source_id':'s1'},{'source_id':'s2'}])
  self.assertEqual(len(rows),1);self.assertEqual(rows[0]['target_claim_ids'],['c1'])
 def test_score_greedy_preserved_and_does_not_pick_low_score(self):
  self.assertIn('score_greedy',METHODS);ESTIMATOR['mode']='heuristic'
  rows=[{'source_id':str(i),'score':i,'source_characters':1,'target_claim_ids':['c'],'doubt_id':'d'} for i in range(5)]
  state={'catalog':{str(i):{'text':str(i)}for i in range(5)},'claims':{'c':{}}}
  a=allocate_general({},state,rows,'score_greedy');self.assertEqual({q for _,q in a['actions']},{2,3,4})
 def test_input_guard_rejects_shared_source_across_splits(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d);tasks=[{'id':str(i),'kind':'qa','metadata':{'native_id':str(i),'acquisition_partition':part},'source_catalog':{'s':{'text':'same'}}}for i,part in enumerate(['fit','diagnostic'])]
   (p/'tasks.json').write_text(json.dumps(tasks));(p/'split_commitment.json').write_text(json.dumps({'task_split':{'0':'fit','1':'diagnostic'}}))
   with self.assertRaisesRegex(ValueError,'shared source'):validate_inputs(p/'tasks.json')
