import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from experiments.consensus_recommendation.common import fingerprint
from experiments.prepare_protocol_v3 import build
from experiments.evidence_consensus import protocol_v3 as p
from experiments.evidence_consensus.allocation_v3 import Allocation
from experiments.evidence_consensus.runner_protocol_v3 import task_case,prepare,select,score

def task(kind='qa',native='n1'):
 t={'id':'test:'+kind+':'+native,'kind':kind,'prompt':'Does alpha have a documented property?','options':['yes','no']if kind=='qa'else [],'candidates':[{'movie_id':str(i),'title':'Alpha '+str(i)}for i in range(15)]if kind=='rec'else [],'initial':'abstain'if kind=='qa'else [str(i)for i in range(15)],'metadata':{'native_id':native,'acquisition_partition':'diagnostic','protocol_v3':{'initial_source_ids':['s0','s1','s2'],'additional_action_cap':2}},'source_catalog':{}}
 for i in range(8):t['source_catalog']['s'+str(i)]={'source_id':'s'+str(i),'kind':'cached_web_reference'if kind=='qa'else 'training_aggregate','title':'Alpha','text':'Alpha documented property number '+str(i)+'.','scope':{'task_id':t['id'],'subject':'external_reference'},'retrieval_score_v3':.2,'allowed_uses':['prediction_basis']}
 return t
class FakeAPI:
 profile={'input_cny_per_million':.15,'output_cny_per_million':1.5}
 def __init__(self,root):self.root=root;self.requests=[]
 def call(self,slot,req):
  payload=json.loads(req['messages'][1]['content']);self.requests.append(payload);cid=payload['claim_id_to_use'];ids=list(payload['fetched_sources']);answer='yes'if payload['task_kind']=='qa'else [str(i)for i in range(10)]
  r={'A':[{'id':cid,'text':'Alpha has a documented property.','scope':{'task_id':payload['task_id']}}],'E':ids[:1],'I':[{'claim_id':cid,'source_id':sid,'relation':'unclear'}for sid in ids[:1]],'decision':answer if payload['phase']=='initial'else payload['current_decision'],'C':{'proposed_patch':{'expected_version':payload['current_version'],'operation':'abstain','evidence_ids':[],'preserve_ids':[],'value':None,'a':None,'b':None}},'findings':[],'proposed_source_ids':[]}
  if payload['phase']=='review':
   a=payload['assigned_review'];sid=a['assigned_source_id'];r['findings']=[{'target_claim_id':c,'source_id':sid,'verdict':'unclear','quote':payload['fetched_sources'][sid]['text'][:15],'reason':'Insufficient for a change.'}for c in a['target_claim_ids']]
  return {'report':r,'model':'fake','usage':{'prompt_tokens':100,'completion_tokens':100}}
class ProtocolTests(unittest.TestCase):
 def payload(self):
  t=task();s=p.initial_state(t,0);s['claims']['c']={'id':'c','text':'Alpha is documented.','scope':{'task_id':t['id']}}
  row={'source_id':'s4','target_claim_ids':['c'],'question':'verify alpha','doubt_id':'d'};p.fetch_source(s,'s4');req=p.request(t,s,'review',0,(),p.assignment(row));return t,s,req,json.loads(req['messages'][1]['content'])
 def test_no_answer_examples_and_explicit_source(self):
  _,_,req,payload=self.payload();self.assertNotIn('schema_example',payload);self.assertNotIn('non_abstain_shape_only_not_a_recommendation',payload);self.assertEqual(payload['assigned_review']['assigned_source_id'],'s4');self.assertNotIn(p.PLACEHOLDER,json.dumps(req))
 def test_irrelevant_material_can_abstain(self):
  t,s,req,payload=self.payload()
  with tempfile.TemporaryDirectory()as d:r=FakeAPI(Path(d)).call('x',req)['report']
  p.check_report(r,payload)
 def test_no_repair_of_fabricated_quote(self):
  t,s,req,payload=self.payload()
  with tempfile.TemporaryDirectory()as d:r=FakeAPI(Path(d)).call('x',req)['report']
  r['findings'][0]['quote']='invented quotation'
  with self.assertRaisesRegex(ValueError,'literal'):p.check_report(r,payload)
 def test_placeholder_rejected(self):
  _,_,req,payload=self.payload()
  with tempfile.TemporaryDirectory()as d:r=FakeAPI(Path(d)).call('x',req)['report']
  r['A'][0]['text']=p.PLACEHOLDER
  with self.assertRaisesRegex(ValueError,'placeholder'):p.check_report(r,payload)
 def test_unassigned_target_rejected(self):
  _,_,req,payload=self.payload()
  with tempfile.TemporaryDirectory()as d:r=FakeAPI(Path(d)).call('x',req)['report']
  r['findings'][0]['target_claim_id']='unassigned'
  with self.assertRaises(ValueError):p.check_report(r,payload)
 def test_nonabstain_old_source_rejected(self):
  _,_,req,payload=self.payload()
  with tempfile.TemporaryDirectory()as d:r=FakeAPI(Path(d)).call('x',req)['report']
  r['C']['proposed_patch'].update(operation='set_answer',value='no',evidence_ids=['s0'])
  with self.assertRaisesRegex(ValueError,'assigned_source_not_used'):p.check_report(r,payload)
 def test_preparation_does_not_transfer_conditional_scores(self):
  ts=[task('qa'),task('rec','1:1'),task('rec','2:1')];before=copy.deepcopy(ts);out,c,d=build(ts)
  self.assertEqual(ts,before)
  for t in out:
   self.assertEqual(len(t['source_catalog']),8)
   for s in t['source_catalog'].values():
    if t['kind']=='qa':self.assertEqual(s['kind'],'cached_web_reference')
    else:self.assertEqual(s['conditioned_on_native_id'],t['metadata']['native_id'])
 def test_features_not_task_or_source_identity(self):
  t=task();s=p.initial_state(t,0);r={'source_id':'s1','target_claim_ids':['x']};fs=p.features(t,s,r,[],0)
  self.assertTrue(any(k.startswith('source_kind:')for k in fs));self.assertNotIn(t['id'],str(fs));self.assertIn('retrieval_score',fs)
 def test_six_arm_execution_privacy_and_budget(self):
  for kind in ['qa','rec']:
   with self.subTest(kind=kind),tempfile.TemporaryDirectory()as d:
    root=Path(d);api=FakeAPI(root);t=task(kind);case=task_case(0,t,api,root)
    self.assertEqual(len(case['methods']),6)
    for a in case['methods'].values():self.assertEqual(a['logical_calls'],5);self.assertTrue(all(e['valid']for e in a['events']))
    initial=[x for x in api.requests if x['phase']=='initial'];self.assertEqual([len(x['fetched_sources'])for x in initial[:3]],[1,1,1]);self.assertEqual([len(x['fetched_sources'])for x in initial[3:]],[1,2,3])
    independent=[x for x in api.requests if x['phase']=='review'and len(x['own_or_permitted_reports'])==1];self.assertEqual(len(independent),2)
    for x in independent:self.assertEqual(len(x['current_claims']),1)
    count=len(api.requests);self.assertEqual(task_case(0,t,api,root),case);self.assertEqual(len(api.requests),count)
 def test_wrong_model_and_train_overlap_blocked(self):
  import joblib
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);t=task();inp=root/'tasks.json';inp.write_text(json.dumps([t]));(root/'split_commitment.json').write_text(json.dumps({'task_split':{t['id']:'diagnostic'}}));model=root/'model.joblib';joblib.dump({'feature_schema':'old'},model)
   with self.assertRaisesRegex(ValueError,'schema'):prepare(root/'out',inp,'learned',model)
   joblib.dump({'feature_schema':'v3-query-action-reviewer-v1'},model)
   with self.assertRaisesRegex(ValueError,'overlap'):prepare(root/'out',inp,'learned',model)
class GameTests(unittest.TestCase):
 def actions(self):return [{'id':str(i),'unit':str(i),'targets':['c'],'gains':[.1+i*.01]*3,'cost':.01}for i in range(5)]
 def test_shared_cap_and_no_duplicate_source(self):
  g=Allocation(self.actions());self.assertFalse(g.feasible((0,0,None)));self.assertFalse(g.feasible((0,1,2)));self.assertTrue(g.feasible((0,1,None)))
 def test_equilibrium_and_central_bound(self):
  g=Allocation(self.actions());s=g.solve();self.assertLessEqual(s['game_potential'],s['central_potential']+1e-10);self.assertLessEqual(s['max_feasible_deviation'],1e-10)
  for row in s['trace']:self.assertGreater(row['delta'],0)
 def test_marginal_utility_equals_potential_difference(self):
  g=Allocation(self.actions());s=(0,1,None);self.assertAlmostEqual(g.utility(s,0),g.potential(s)-g.potential((None,1,None)))
 def test_abstain_when_negative_net_value(self):
  a=self.actions()
  for row in a:row['gains']=[0]*3
  self.assertEqual(Allocation(a).solve()['game'],(None,None,None))
 def test_cost_budget(self):
  a=self.actions();g=Allocation(a,budget=.009);self.assertEqual(g.solve()['central'],(None,None,None))
if __name__=='__main__':unittest.main()

class EndToEndTests(unittest.TestCase):
 def test_acquire_fit_compare_report_and_resume_without_api(self):
  import contextlib,io
  from experiments.acquire_protocol_v3 import prepare as prep_train,collect,gate
  from experiments.fit_protocol_v3 import fit
  from experiments.report_protocol_v3 import report
  from experiments.consensus_recommendation.common import create
  with tempfile.TemporaryDirectory()as d,contextlib.redirect_stdout(io.StringIO()):
   root=Path(d);inputs=root/'inputs';inputs.mkdir();train=[]
   for kind in ['qa','rec']:
    for i in range(3):
     t=task(kind,'train'+str(i));t['metadata']['acquisition_partition']='fit'if i<2 else 'calibration';train.append(t)
   inp=inputs/'tasks.json';create(inp,train);create(inputs/'split_commitment.json',{'task_split':{t['id']:t['metadata']['acquisition_partition']for t in train}})
   trainroot=root/'acquisition';tasks,plan=prep_train(trainroot,inp);api=FakeAPI(trainroot);cases=[collect(i,t,trainroot,api)for i,t in enumerate(tasks)];self.assertTrue(gate(cases)['passed'])
   create(trainroot/'generation_complete.json',{'task_ids':[t['id']for t in tasks],'hashes':[fingerprint(c)for c in cases]})
   labels={t['id']:({'kind':'qa','answer':'yes'}if t['kind']=='qa'else {'kind':'rec','targets':['0','11']})for t in tasks};create(trainroot/'labels.after_generation.json',labels);fit(trainroot,root/'model');fit(trainroot,root/'model')
   diag=[task('qa','heldqa'),task('rec','heldrec')];di=root/'diag';create(di/'tasks.json',diag);create(di/'split_commitment.json',{'task_split':{t['id']:'diagnostic'for t in diag}})
   labels={t['id']:({'kind':'qa','answer':'yes'}if t['kind']=='qa'else {'kind':'rec','targets':['0','11']})for t in diag}
   for mode in ('heuristic','learned'):
    runroot=root/mode;tasks,plan,bundle=prepare(runroot,di/'tasks.json',mode,root/'model/model.joblib'if mode=='learned'else None);api=FakeAPI(runroot);cases=[task_case(i,t,api,runroot,bundle)for i,t in enumerate(tasks)];create(runroot/'generation_complete.json',{'task_ids':[t['id']for t in tasks],'hashes':[fingerprint(c)for c in cases]})
    with patch('experiments.evidence_consensus.runner_protocol_v3.load_evaluation_labels',return_value=labels):s=score(runroot)
    self.assertEqual(s['methods']['game']['qa_tasks'],1);self.assertEqual(s['methods']['game']['rec_targets'],2)
   create(root/'inputs/excluded_tasks.json',[])
   with patch('experiments.evidence_consensus.runner_protocol_v3.load_evaluation_labels',return_value=labels):report(root);report(root)
   self.assertTrue((root/'report.md').exists())
 def test_real_input_retrieval_and_split_qualification(self):
  from experiments.consensus_recommendation.common import ROOT
  src=ROOT/'experiments/results/e2_action_acquisition_160_live_v1/inputs.json'
  if not src.exists():self.skipTest('local cached development data absent')
  tasks=json.loads(src.read_text());out,commit,diagnostics=build(tasks)
  self.assertEqual(len(out),160);self.assertEqual(sum(d['budget_pressure']for d in diagnostics),143)
  self.assertTrue(all(s['kind']=='cached_web_reference'for t in out if t['kind']=='qa'for s in t['source_catalog'].values()))
  for t in out:
   if t['kind']=='rec':self.assertTrue(all(s['conditioned_on_native_id']==t['metadata']['native_id']for s in t['source_catalog'].values()))
 def test_target_specific_edge_required_for_non_unclear_finding(self):
  helper=ProtocolTests();_,_,req,payload=helper.payload()
  with tempfile.TemporaryDirectory()as d:r=FakeAPI(Path(d)).call('x',req)['report']
  r['findings'][0]['verdict']='support'
  with self.assertRaisesRegex(ValueError,'target-specific'):p.check_report(r,payload)
 def test_current_version_contract(self):
  helper=ProtocolTests();_,_,req,payload=helper.payload()
  with tempfile.TemporaryDirectory()as d:r=FakeAPI(Path(d)).call('x',req)['report']
  r['C']['proposed_patch']['expected_version']=99
  with self.assertRaisesRegex(ValueError,'current-version'):p.check_report(r,payload)

class UpdateAndStressTests(unittest.TestCase):
 def test_positive_local_update_preserves_other_positions_and_records_target(self):
  from experiments.evidence_consensus.state import initialize_decision,register_report
  class UpdatingAPI(FakeAPI):
   def call(self,slot,req):
    r=super().call(slot,req);x=json.loads(req['messages'][1]['content']);a=x['assigned_review'];sid=a['assigned_source_id'];target=a['target_claim_ids'][0];report=r['report'];report['E']=[sid];report['I']=[{'claim_id':target,'source_id':sid,'relation':'refute'}]
    report['findings']=[{'target_claim_id':target,'source_id':sid,'verdict':'refute','quote':x['fetched_sources'][sid]['text'][:20],'reason':'Evidence contradicts the target hypothesis.'}]
    report['C']['proposed_patch'].update(evidence_ids=[sid],operation='set_answer'if x['task_kind']=='qa'else 'swap_candidate',value='yes'if x['task_kind']=='qa'else None,a=None if x['task_kind']=='qa'else '0',b=None if x['task_kind']=='qa'else '12');return r
  for kind in ['qa','rec']:
   with self.subTest(kind=kind),tempfile.TemporaryDirectory()as d:
    t=task(kind);state=p.initial_state(t,0);state=initialize_decision(state,'no'if kind=='qa'else t['initial']);state['claims']['c']={'id':'c','text':'Alpha lacks the property.','scope':{'task_id':t['id']},'version':0,'status':'unverified'};before=copy.deepcopy(state)
    row={'source_id':'s4','target_claim_ids':['c'],'doubt_id':'d','question':'verify c'};after,event=p.review(UpdatingAPI(Path(d)),'positive',t,state,row,[],0,__import__('experiments.evidence_consensus.runner_protocol_v3',fromlist=['ledger']).ledger())
    self.assertTrue(event['patch_accepted']);self.assertFalse(after.get('semantic_truth_certified',False));self.assertEqual(after['claims']['c'],before['claims']['c']);self.assertEqual(after['claim_assessments'][0]['target_claim_id'],'c')
    if kind=='qa':self.assertEqual(after['decision'],'yes')
    else:
     self.assertEqual(after['decision'][0],'12');self.assertEqual(after['decision'][1:10],before['decision'][1:10]);self.assertEqual(set(after['decision']),set(before['decision']))
 def test_real_stress_variants_preserve_tasks_and_initial_allocation(self):
  from experiments.prepare_protocol_v3_stress import variants
  from experiments.consensus_recommendation.common import ROOT
  path=ROOT/'experiments/results/e2_action_acquisition_160_live_v1/inputs.json'
  if not path.exists():self.skipTest('cached inputs absent')
  original=json.loads(path.read_text());tasks,_,_=build(original);di=[t for t in tasks if t['metadata']['acquisition_partition']=='diagnostic'and len(t['source_catalog'])>5];before=copy.deepcopy(di);out=variants(di,original);self.assertEqual(di,before)
  for name,rows in out.items():
   self.assertEqual([t['id']for t in rows],[t['id']for t in di])
   for a,b in zip(di,rows):
    self.assertEqual(a['prompt'],b['prompt']);self.assertEqual(a['candidates'],b['candidates']);self.assertEqual(a['metadata']['protocol_v3']['initial_source_ids'],b['metadata']['protocol_v3']['initial_source_ids'])
    for sid in a['metadata']['protocol_v3']['initial_source_ids']:self.assertEqual(a['source_catalog'][sid],b['source_catalog'][sid])
    if name=='missing':self.assertEqual(len(b['source_catalog']),len(a['source_catalog'])-1)
    if name=='redundant':self.assertEqual(len({s['text']for s in b['source_catalog'].values()}),len({s['text']for s in a['source_catalog'].values()}))
 def test_grouped_paired_interval(self):
  from experiments.evidence_consensus.paired_statistics_v3 import paired
  t=task();labels={t['id']:{'kind':'qa','answer':'yes'}};c={'task_id':t['id'],'kind':'qa','methods':{'game':{'decision':'yes'},'independent':{'decision':'no'}}};s=paired([t],[c],labels,resamples=100)
  self.assertEqual(s['qa']['difference'],1);self.assertEqual(s['qa']['cluster_bootstrap_95_percent'],[1,1]);self.assertEqual(s['qa']['exact_mcnemar_p_unadjusted'],1.)

class PublicCorpusTests(unittest.TestCase):
 def test_explicit_public_corpus_replaces_small_cache_without_labels(self):
  import sqlite3
  with tempfile.TemporaryDirectory()as d:
   path=Path(d)/'corpus.sqlite';con=sqlite3.connect(path);con.execute('create table state (k text,v text)');con.execute("insert into state values ('count','12')");con.execute('create table docs(docid text,title text,text text)');con.execute('create virtual table docs_fts using fts5(title,text)')
   for i in range(12):
    title='Alpha '+str(i);text='Alpha documented property in public source '+str(i);con.execute('insert into docs values (?,?,?)',(str(i),title,text));con.execute('insert into docs_fts values (?,?)',(title,text))
   con.commit();con.close();out,commit,diagnostics=build([task()],corpus_path=path)
   self.assertEqual(len(out[0]['source_catalog']),8);self.assertTrue(diagnostics[0]['budget_pressure']);self.assertTrue(all(s['retrieval_engine']=='title-anchor + BM25 public corpus'for s in out[0]['source_catalog'].values()));self.assertFalse(commit['label_access'])
