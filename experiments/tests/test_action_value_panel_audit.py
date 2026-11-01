import json,tempfile,unittest
from pathlib import Path
from experiments.audits.audit_action_value_panel import audit,decompose

def branch(sources,rep,valid=True,accepted=False,reason='explicit_abstention',answer='abstain',error=None):
 return {'source_ids':list(sources),'replicate':rep,'valid':valid,'patch_accepted':accepted,'patch_reason':reason,'after_decision':answer,'error':error}

def qa_case(task_id,patched_answer=None):
 sources=['s1','s2'];branches=[branch((),0),branch((),1),branch(['s1'],0,accepted=patched_answer is not None,reason='operationally_valid'if patched_answer else 'explicit_abstention',answer=patched_answer or 'abstain'),branch(['s1'],1),branch(['s2'],0),branch(['s2'],1),branch(sources,0),branch(sources,1,valid=False,error='quote not in actually visible source')]
 return {'task_id':task_id,'kind':'qa','baseline':{'decision':'abstain'},'predictions':{'s1':{'gain':.1},'s2':{'gain':.2}},'branches':branches}

class ActionValuePanelAuditTests(unittest.TestCase):
 def test_decompose_reports_task_local_changes_and_zero_gains(self):
  labels={'t1':{'kind':'qa','answer':'yes'},'t2':{'kind':'qa','answer':'yes'}}
  d=decompose([qa_case('t1','no'),qa_case('t2')],labels);st=d['kinds']['qa']
  self.assertEqual(st['branches'],16);self.assertEqual(st['invalid_branches'],2)
  self.assertEqual(st['valid_outcomes']['False|explicit_abstention'],13);self.assertEqual(st['valid_outcomes']['True|operationally_valid'],1)
  self.assertEqual(st['decision_changes'],1);self.assertEqual(st['decision_changes_without_new_evidence'],0);self.assertEqual(st['decision_changes_to_gold'],0);self.assertEqual(st['decision_change_answers'],{'"no"':1})
  self.assertEqual(d['per_task'][0]['decision_changes'],1);self.assertEqual(d['per_task'][1]['decision_changes'],0)
  rec=st['recomputed']
  self.assertAlmostEqual(rec['initial_quality_sum'],0.);self.assertAlmostEqual(rec['predicted_choice_gain_sum'],0.);self.assertAlmostEqual(rec['uniform_choice_gain_sum'],0.);self.assertAlmostEqual(rec['single_repeat_absolute_difference_sum'],0.)
  self.assertEqual(rec['tasks_with_positive_single_mean'],0);self.assertEqual(rec['tasks_with_varying_single_means'],0);self.assertEqual(rec['pairs_joint_only_in_all_repeats_valid'],0);self.assertEqual(rec['branches'],16);self.assertEqual(rec['invalid_branches'],2)

 def test_audit_rejects_seal_mismatch(self):
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp);(root/'cases').mkdir()
   (root/'inputs.json').write_text(json.dumps([{'id':'t1'}]));(root/'cases'/'case_0.json').write_text(json.dumps(qa_case('t1')))
   (root/'generation_complete.json').write_text(json.dumps({'task_ids':['other'],'hashes':['x']}))
   for name in('plan.json','summary.json','physical_summary.json','labels.after_generation.json'):(root/name).write_text('{}')
   with self.assertRaises(ValueError):audit(root)

if __name__=='__main__':unittest.main()
