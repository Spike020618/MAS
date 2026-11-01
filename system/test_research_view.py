import json,tempfile,unittest
from pathlib import Path
from system.game_consensus_recsys.research_view import inventory,sealed_case
from experiments.consensus_recommendation.common import fingerprint
class ReplayTests(unittest.TestCase):
 def test_unsealed_cases_and_arbitrary_stage_not_exposed(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);self.assertFalse(inventory(root)['stages'][0]['sealed'])
   with self.assertRaises(RuntimeError):sealed_case('main',0,root)
   with self.assertRaises(ValueError):sealed_case('../../.env.local',0,root)
 def test_only_integrity_checked_case_and_task_fields_returned(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);p=root/'main';(p/'cases').mkdir(parents=True);case={'task_id':'t','methods':{},'labels_available':False}
   task={'id':'t','kind':'rec','prompt':'Q','candidates':[],'source_catalog':{},'metadata':{'evaluation_vault':'must_not_expose'}}
   (p/'cases/case_0.json').write_text(json.dumps(case));(p/'inputs.json').write_text(json.dumps([task]));(p/'generation_complete.json').write_text(json.dumps({'task_ids':['t'],'hashes':[fingerprint(case)]}))
   out=sealed_case('main',0,root);self.assertNotIn('metadata',out['task']);self.assertFalse(out['labels_exposed'])
   case['methods']={'fake':{}};(p/'cases/case_0.json').write_text(json.dumps(case))
   with self.assertRaises(RuntimeError):sealed_case('main',0,root)
if __name__=='__main__':unittest.main()
