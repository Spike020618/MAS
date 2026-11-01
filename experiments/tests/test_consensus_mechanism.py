import copy,contextlib,io,itertools,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from experiments.tests.test_protocol_v3 import task
from experiments.consensus_mechanism import core as c
from experiments.consensus_mechanism.allocation import Allocator
from experiments.consensus_mechanism.engine import initialize,execute,PRIMARY,EXTERNAL,ABLATIONS
from experiments.consensus_mechanism.runner import training_case,fit,eval_case,score
from experiments.consensus_recommendation.common import create,fingerprint

class API:
 def __init__(self,root):self.root=root;self.requests=[]
 def call(self,slot,req):
  x=json.loads(req['messages'][1]['content']);self.requests.append(x);phase=x['phase'];sp=list(x['evidence_spans']);qa=x['task_type']=='qa'
  if phase in ('initial','direct'):
   decision=('no'if phase=='initial'else'yes')if qa else([str(i)for i in range(10)]if phase=='initial'else ['11']+[str(i)for i in range(1,10)])
   report={'decision':decision,'claim':'Alpha lacks a documented property.','span_ids':sp[:1]}
  elif phase=='feedback':report={'critique':'Check the alpha evidence.','span_ids':sp[:1]}
  else:
   action=x['assigned_action'];extra=action['source_ids'];ids=[k for k,v in x['evidence_spans'].items()if v['source_id']in extra]
   report={'target_id':action['target_id'],'verdict':'refute'if ids else'unclear','span_ids':ids[:1],'rationale':'The assigned evidence addresses the premise.','patch':({'operation':'set_answer','value':'yes'}if qa else {'operation':'swap_candidate','a':x['current_decision'][0],'b':'11'})if ids and (qa or '11'not in x['current_decision'])else None}
  return {'report':report,'model':'fixture','usage':{'prompt_tokens':100,'completion_tokens':100}}

class CoreTests(unittest.TestCase):
 def test_spans_are_exact_source_substrings(self):
  t=task();pool=c.spans(t)
  for span in pool.values():self.assertEqual(span['text'],t['source_catalog'][span['source_id']]['text'][span['start']:span['end']])
 def test_initial_prompt_has_no_review_instruction_or_prior_decision(self):
  t=task();s=c.fresh(t);c.expose(t,s,['s0']);req=c.request(t,s,'initial',0);x=json.loads(req['messages'][1]['content']);self.assertNotIn('current_decision',x);self.assertNotIn('target_id',x);self.assertNotIn('decision MUST equal',req['messages'][0]['content'])
 def test_unseen_or_fabricated_span_rejected(self):
  t=task();s=c.expose(t,c.fresh(t),['s0']);r={'decision':'yes','claim':'Alpha exists.','span_ids':[next(k for k,v in c.spans(t).items()if v['source_id']=='s1')]}
  with self.assertRaisesRegex(ValueError,'exposed'):c.validate_report(t,s,r,'initial',0)
 def test_transactional_patch_preserves_unrelated_ranking(self):
  with tempfile.TemporaryDirectory()as d:
   t=task('rec');api=API(Path(d));s,reports,events,_=initialize(t,0,api);before=copy.deepcopy(s);c.expose(t,s,['s4']);after,e=c.invoke(api,'review',t,s,'review',3,reports,{'target_id':'claim:0','source_ids':['s4']})
   self.assertTrue(e['valid']);self.assertTrue(e['accepted']);self.assertEqual(after['decision'][1:10],before['decision'][1:10]);self.assertEqual(before['decision'][0],'0');self.assertEqual(after['decision'][0],'11');self.assertEqual(after['claims']['claim:0']['status'],'model_refuted')
 def test_invalid_output_keeps_exact_state(self):
  class Bad(API):
   def call(self,slot,req):return {'report':{},'model':'fixture','usage':{}}
  t=task();s=c.fresh(t);before=copy.deepcopy(s)
  with tempfile.TemporaryDirectory()as d:after,e=c.invoke(Bad(Path(d)),'bad',t,s,'initial',0)
  self.assertEqual(s,before);self.assertEqual(after,before);self.assertFalse(e['valid'])
 def test_all_primary_external_and_ablations_obey_call_budget(self):
  for kind in ('qa','rec'):
   with tempfile.TemporaryDirectory()as d:
    t=task(kind);api=API(Path(d));base,reports,initial,private=initialize(t,0,api)
    for method in PRIMARY+EXTERNAL+ABLATIONS:
     arm=execute(t,0,method,api,base,reports,initial,private,budget=2)
     self.assertLessEqual(arm['logical_calls'],5,method);self.assertTrue(all(e['valid']for e in arm['events']),(method,[e.get('error')for e in arm['events']if not e['valid']]))
    self.assertEqual(len(base['visible']),3)
 def test_features_have_no_native_identity(self):
  with tempfile.TemporaryDirectory()as d:
   t=task();s,reports,_,_=initialize(t,0,API(Path(d)));a=c.actions(t,s)[0];f=c.features(t,s,a,reports,0);self.assertNotIn(t['id'],str(f));self.assertNotIn(t['metadata']['native_id'],str(list(f)))

class AllocationTests(unittest.TestCase):
 def rows(self):return [{'id':str(i),'source_ids':[str(i)],'gains':[.1+i*.01]*3,'cost':.001}for i in range(5)]
 def test_joint_internal_complement_can_be_selected(self):
  rows=[{'id':'a','source_ids':['a'],'gains':[0]*3,'cost':.01},{'id':'b','source_ids':['b'],'gains':[0]*3,'cost':.01},{'id':'ab','source_ids':['a','b'],'gains':[.5]*3,'cost':.01}];a=Allocator(rows);s=a.solve('game');self.assertIn(2,s['assignment']);self.assertTrue(a.feasible(s['assignment']));self.assertLessEqual(s['max_feasible_deviation'],1e-10)
 def test_source_call_and_money_bounds(self):
  a=Allocator(self.rows());self.assertFalse(a.feasible((0,0,None)));self.assertFalse(a.feasible((0,1,2)));self.assertTrue(a.feasible((0,1,None)));self.assertFalse(Allocator(self.rows(),money=0).feasible((0,None,None)))
 def test_exact_potential_and_central_bound(self):
  a=Allocator(self.rows());s=a.solve('game');central=a.solve('central');self.assertLessEqual(s['potential'],central['potential']+1e-10)
  state=(0,1,None)
  for i in range(3):
   for t in a.deviations(state,i):self.assertAlmostEqual(a.utility(t,i)-a.utility(state,i),a.potential(t)-a.potential(state))
 def test_no_false_gain_when_estimates_negative(self):
  rows=self.rows()
  for r in rows:r['gains']=[-.1]*3
  self.assertEqual(Allocator(rows).solve('game')['assignment'],(None,None,None))

class PipelineTests(unittest.TestCase):
 def test_training_evaluation_and_resume(self):
  import joblib
  with tempfile.TemporaryDirectory()as d,contextlib.redirect_stdout(io.StringIO()):
   root=Path(d);ts=[]
   for i in range(6):
    t=task('qa'if i%2==0 else'rec','train'+str(i));t['metadata']['acquisition_partition']='fit'if i<4 else'calibration';ts.append(t)
   train=root/'train';cases=[training_case(i,t,train,API(train))for i,t in enumerate(ts)];create(train/'inputs.json',ts);create(train/'generation_complete.json',{'task_ids':[t['id']for t in ts],'hashes':[fingerprint(x)for x in cases]});gold={t['id']:({'kind':'qa','answer':'yes'}if t['kind']=='qa'else {'kind':'rec','targets':['11']})for t in ts}
   with patch('experiments.consensus_mechanism.runner.labels_for',return_value=gold):fit(train,root/'model');fit(train,root/'model')
   bundle=joblib.load(root/'model/model.joblib');t=task('qa','held');out=root/'eval';api=API(out);case=eval_case(0,t,out,api,PRIMARY+EXTERNAL,2,bundle);count=len(api.requests);self.assertEqual(eval_case(0,t,out,api,PRIMARY+EXTERNAL,2,bundle),case);self.assertEqual(len(api.requests),count)
   create(out/'inputs.json',[t]);create(out/'plan.json',{'methods':list(PRIMARY+EXTERNAL),'provider':'qwen_flash','scope':'fixture','additional_calls':2});create(out/'generation_complete.json',{'task_ids':[t['id']],'hashes':[fingerprint(case)]})
   with patch('experiments.consensus_mechanism.runner.labels_for',return_value={t['id']:{'kind':'qa','answer':'yes'}}):s=score(out);score(out)
   self.assertEqual(s['tasks'],1);self.assertEqual(s['methods']['game']['qa_correct'],1)
if __name__=='__main__':unittest.main()
