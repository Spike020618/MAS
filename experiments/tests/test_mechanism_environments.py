import copy,json,tempfile,unittest
from pathlib import Path
from experiments.prepare_mechanism_environments import build,prepare,ENVIRONMENTS
class EnvironmentTests(unittest.TestCase):
 def tasks(self):
  return [{'id':str(i),'kind':'rec','metadata':{'native_id':'c:'+str(i)},'prompt':'question','candidates':['1','2'],'initial':['1','2'],'source_catalog':{'s':{'source_id':'s','text':'real text','scope':{'task_id':str(i)}}}}for i in range(2)]
 def test_no_mutation_or_candidate_injection(self):
  tasks=self.tasks();before=copy.deepcopy(tasks)
  for env in ENVIRONMENTS:
   rows=build(tasks,env)
   for a,b in zip(tasks,rows):self.assertEqual(a['candidates'],b['candidates']);self.assertEqual(a['initial'],b['initial']);self.assertEqual(a['id'],b['id'])
  self.assertEqual(tasks,before)
 def test_duplicate_is_one_evidence_unit(self):
  row=build(self.tasks(),'redundant')[0];self.assertEqual(len(row['source_catalog']),1);self.assertEqual(row['source_catalog']['s']['independent_evidence_units'],1)
 def test_all_variants_and_conversation_share_split(self):
  with tempfile.TemporaryDirectory()as d:
   inp=Path(d)/'input.json';inp.write_text(json.dumps(self.tasks()));out=Path(d)/'out';prepare(inp,out)
   splits=[]
   for env in ENVIRONMENTS:splits.extend(x['metadata']['group_split']for x in json.loads((out/env/'tasks.json').read_text()))
   self.assertEqual(len(set(splits)),1)
 def test_foreign_material_never_claims_current_fact_authority(self):
  x=build(self.tasks(),'distractor')[0]
  for k,v in x['source_catalog'].items():
   if k.startswith('distractor:'):self.assertFalse(v['current_user_fact_authority']);self.assertEqual(v['scope']['task_id'],x['id'])

class ExposureVisibilityTests(unittest.TestCase):
 def test_both_copies_survive_actual_runner_bound(self):
  from experiments.evidence_consensus.runner_v6 import bounded_task
  t=EnvironmentTests().tasks()[0];t['source_catalog']['s']['text']='x'*1700
  baseline=build([t],'baseline')[0];repeated=build([t],'redundant')[0]
  self.assertEqual(len(baseline['source_catalog']['s']['text']),1000)
  visible=bounded_task(repeated)['source_catalog']['s']['text']
  self.assertEqual(visible,'x'*1000+'\n\n'+'x'*1000)
  self.assertEqual(repeated['source_catalog']['s']['canonical_original_content_hash'],baseline['source_catalog']['s']['canonical_original_content_hash'])
  self.assertEqual(repeated['source_catalog']['s']['independent_evidence_units'],1)
