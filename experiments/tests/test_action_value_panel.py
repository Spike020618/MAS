import copy,contextlib,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from experiments.tests.test_protocol_v3 import task,FakeAPI
from experiments.action_value_panel import subsets,task_case,analyze,check,request,prepare,score
from experiments.consensus_recommendation.common import create,fingerprint
from experiments.evidence_consensus import protocol_v3 as p

class PanelAPI(FakeAPI):
 def call(self,slot,req):
  x=json.loads(req['messages'][1]['content'])
  if req['kind']!='matched_state_panel_review':
   r=super().call(slot,req)
   if x['task_kind']=='qa':r['report']['decision']='no'
   return r
  self.requests.append(x);extra=x['additional_source_ids'];pair=len(extra)==2
  evidence=[{'source_id':sid,'quote':x['fetched_sources'][sid]['text'][:20],'role':'context','reason':'Partial premise for the combined argument.'}for sid in extra]if pair else []
  report={'target_claim_id':x['target_claim_id'],'target_verdict':'support'if pair else 'unclear','evidence':evidence,'rationale':'A joint argument.'if pair else 'Insufficient evidence.','patch':{'expected_version':x['current_version'],'operation':('set_answer'if x['task_kind']=='qa'else 'swap_candidate')if pair else 'abstain','evidence_ids':extra if pair else [],'preserve_ids':[],'value':'yes'if pair and x['task_kind']=='qa'else None,'a':'0'if pair and x['task_kind']=='rec'else None,'b':'11'if pair and x['task_kind']=='rec'else None}}
  return {'report':report,'model':'fake','usage':{'prompt_tokens':100,'completion_tokens':100}}

class PanelTests(unittest.TestCase):
 def case(self,t,root):
  api=PanelAPI(root)
  with contextlib.redirect_stdout(io.StringIO()):c=task_case(0,t,root,api)
  return c,api
 def test_all_subsets_and_equal_repetitions(self):
  self.assertEqual(len(subsets(list('abcde'))),16);self.assertEqual(len(set(subsets(list('abcde')))),16)
  with tempfile.TemporaryDirectory()as d:
   c,api=self.case(task(),Path(d));self.assertEqual(len(c['branches']),32);self.assertEqual(len(api.requests),35)
   self.assertEqual(len({e['baseline_hash']for e in c['branches']}),1)
   for x in api.requests[3:]:self.assertEqual(len(x['fetched_sources']),3+len(x['additional_source_ids']))
 def test_joint_only_signal_and_no_baseline_mutation(self):
  for kind in ('qa','rec'):
   with tempfile.TemporaryDirectory()as d:
    t=task(kind);before=copy.deepcopy(t);c,api=self.case(t,Path(d));label={'kind':'qa','answer':'yes'}if kind=='qa'else {'kind':'rec','targets':['11']};r=analyze([t],[c],{t['id']:label});st=r['kinds'][kind]
    self.assertEqual(t,before);self.assertEqual(len(c['baseline']['visible_sources']),3);self.assertEqual(st['tasks_with_repeat_consistent_joint_only_pair'],1);self.assertEqual(st['pairs_joint_only_in_all_repeats_valid'],10);self.assertEqual(st['predicted_choice_gain_sum'],0)
 def test_failed_single_does_not_create_valid_joint_only_evidence(self):
  with tempfile.TemporaryDirectory()as d:
   t=task();c,_=self.case(t,Path(d))
   for e in c['branches']:
    if len(e['source_ids'])==1:e['valid']=False
   r=analyze([t],[c],{t['id']:{'kind':'qa','answer':'yes'}});self.assertEqual(r['kinds']['qa']['tasks_with_repeat_consistent_joint_only_pair'],0);self.assertEqual(r['kinds']['qa']['invalid_branches'],10)
 def test_resume_does_not_repeat_model_requests(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);t=task();c,api=self.case(t,root);n=len(api.requests);self.assertEqual(task_case(0,t,root,api),c);self.assertEqual(len(api.requests),n)
 def test_no_future_labels_until_all_cases_sealed(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);t=task();c,api=self.case(t,root);create(root/'inputs.json',[t])
   with patch('experiments.action_value_panel.load_evaluation_labels')as labels:
    with self.assertRaises(FileNotFoundError):score(root)
    labels.assert_not_called()
   create(root/'generation_complete.json',{'task_ids':[t['id']],'hashes':[fingerprint(c)]});create(root/'plan.json',{'estimator':'heuristic','model_sha256':None})
   with patch('experiments.action_value_panel.load_evaluation_labels',return_value={t['id']:{'kind':'qa','answer':'yes'}}),contextlib.redirect_stdout(io.StringIO()):r=score(root);score(root)
   self.assertTrue((root/'report.md').exists());self.assertEqual(r['estimator'],'heuristic')
 def test_fabricated_quote_and_partial_pair_citations_rejected(self):
  with tempfile.TemporaryDirectory()as d:
   t=task();c,api=self.case(t,Path(d));e=next(e for e in c['branches']if len(e['source_ids'])==2);state=copy.deepcopy(c['baseline'])
   for sid in e['source_ids']:p.fetch_source(state,sid)
   r=copy.deepcopy(e['report']);r['evidence'][0]['quote']='a completely invented quotation'
   with self.assertRaisesRegex(ValueError,'quote'):check(r,t,state,c['target_claim_id'],e['source_ids'])
   r=copy.deepcopy(e['report']);r['patch']['evidence_ids']=r['patch']['evidence_ids'][:1]
   with self.assertRaisesRegex(ValueError,'all added'):check(r,t,state,c['target_claim_id'],e['source_ids'])
 def test_prediction_and_subset_commitment_before_labels(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);t=task();c,_=self.case(t,root);f=json.loads((root/'predictions/case_0.json').read_text());self.assertFalse(f['labels_available']);self.assertEqual(set(f['predictions']),set(c['predictions']));self.assertFalse((root/'labels.after_generation.json').exists())
if __name__=='__main__':unittest.main()

class SelectionAndRescoreTests(unittest.TestCase):
 def test_preparation_is_label_free_and_has_expected_budget(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);tasks=[task(kind,str(i))for kind in ('qa','rec')for i in range(4)];inputs=root/'tasks.json';create(inputs,tasks)
   with patch('experiments.action_value_panel.load_evaluation_labels')as labels:
    selected,plan,_=prepare(root/'out',inputs);labels.assert_not_called()
   self.assertEqual(len(selected),8);self.assertEqual(plan['physical_cap'],280);self.assertEqual(plan['branches_per_task'],16)
 def test_learned_rescore_uses_sealed_responses_without_calls(self):
  import joblib
  from sklearn.pipeline import make_pipeline
  from sklearn.feature_extraction import DictVectorizer
  from sklearn.dummy import DummyRegressor
  from experiments.action_value_panel import rescore
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);t=task();api=PanelAPI(root)
   with contextlib.redirect_stdout(io.StringIO()):c=task_case(0,t,root,api)
   create(root/'inputs.json',[t]);create(root/'generation_complete.json',{'task_ids':[t['id']],'hashes':[fingerprint(c)]});mroot=root/'model';mroot.mkdir();model=mroot/'model.joblib';gain=make_pipeline(DictVectorizer(),DummyRegressor(strategy='constant',constant=.1)).fit([{'x':0}],[0]);joblib.dump({'gain':gain,'gain_bias':0.,'feature_schema':'v3-query-action-reviewer-v1'},model);create(mroot/'training_tasks.json',[task('qa','other')]);n=len(api.requests)
   with patch('experiments.action_value_panel.load_evaluation_labels',return_value={t['id']:{'kind':'qa','answer':'yes'}}),contextlib.redirect_stdout(io.StringIO()):out=rescore(root,model,root/'rescore')
   self.assertEqual(out['model_api_calls'],0);self.assertEqual(len(api.requests),n);self.assertEqual(json.loads((root/'cases/case_0.json').read_text()),c)
