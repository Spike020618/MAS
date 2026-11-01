import json,unittest
from pathlib import Path
from experiments.coordination_game_v1 import stage3_run as s3

def make_task(i=0):
    return {'id':f'q{i}','question':'Who directed the film A?','sources':[
        {'id':f's{i}{j}','title':f'T{j}','sentences':[f'sentence {j} about A']}for j in range(6)],
        'member_initial_sources':{'0':[f's{i}0'],'1':[f's{i}1'],'2':[f's{i}2']},'source_order':[f's{i}{j}'for j in range(6)],
        'candidates':[f's{i}3',f's{i}4',f's{i}5']}

def fake_estimator(task,member,candidate,packet=None):
    return 1.0 if candidate.endswith('5') else 0.1*member

class Stage3AssignmentTests(unittest.TestCase):
 def test_all_methods_return_budget_pairs(self):
  t=make_task()
  # greedy_adaptive is interleaved in the run loop (needs read responses); the rest are static
  for method in s3.METHODS:
   if method=='greedy_adaptive':continue
   pairs=s3.assignments_for(t,method,3,fake_estimator)
   self.assertEqual(len(pairs),s3.BUDGET_READS)
   for m,c in pairs:self.assertIn(c,t['candidates'])
   if method!='single_concentrated':
    self.assertEqual(len({c for _,c in pairs}),s3.BUDGET_READS)  # no duplicate candidate reads

 def test_random_is_deterministic_by_seed(self):
  t=make_task()
  a=s3.assignments_for(t,'random',7,None);b=s3.assignments_for(t,'random',7,None)
  self.assertEqual(a,b)
  c=s3.assignments_for(t,'random',8,None)
  self.assertNotEqual(a,c)

 def test_estimator_methods_prefer_high_value_candidate(self):
  t=make_task()
  for method in ('greedy','central','game','cbba','single_concentrated'):
   pairs=s3.assignments_for(t,method,0,fake_estimator)
   self.assertTrue(any(c.endswith('5')for _,c in pairs),method)

 def test_question_features_match_stage2_schema(self):
  t=make_task()
  f=s3.question_features(t,0,t['candidates'][0])
  for key in ('member_packet_size','action_claim_coverage','packet_claim_coverage',
              'action_packet_redundancy','action_packet_complement','action_chars','claim_chars'):
   self.assertIn(key,f)
  self.assertNotIn('gold',json.dumps(f))

class Stage3IntegrationTests(unittest.TestCase):
 def test_fake_transport_run_and_analyze(self):
  import tempfile
  from experiments.coordination_game_v1.live_runtime import DurableRuntime
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp);data=root/'data';data.mkdir()
   tasks=[make_task(i)for i in range(4)]
   (data/'tasks.json').write_text(json.dumps(tasks))
   (data/'private').mkdir();(data/'private/scoring.json').write_text(
     json.dumps({t['id']:{'answer':'gold secret answer'}for t in tasks}))
   class FakeT:
    def __call__(self,request):
     content=request['messages'][1]['content']
     if '"readout"' in content:
      return {'http_status':200,'parsed':{'observations':['o'],'citation_ids':['s0'],'answer':'extracted answer'},'usage':{}}
     return {'http_status':200,'parsed':{'observations':['obs'],'citation_ids':['s0'],'answer':'x'},'usage':{}}
   s3.DurableRuntime=lambda out,plan,transport,run,approved_cny:DurableRuntime(out,plan,FakeT(),run=True,approved_cny=5.0)
   out=root/'collect'
   result=s3.run(data,out,fit_report={'direction_model':{'features':['action_claim_coverage'],'weights':[1.0,0.1]}},approved_cny=5.0)
   self.assertEqual(result['records'],4*len(s3.METHODS))
   panel=json.loads((out/'panel.json').read_text())
   self.assertEqual(panel['seal']['budget_violations'],0)
   self.assertTrue((out/'generation_complete.json').exists())
   # every method answered via the extractive readout
   for r in panel['records']:
    self.assertEqual(r['answer'],'extracted answer')
    self.assertTrue(r['valid'])
   # labels never in slots
   for p in (out/'slots').glob('*.json'):
    self.assertNotIn('gold secret answer',p.read_text())
   analysis=s3.analyze(out,data)
   self.assertEqual(set(analysis['methods']),set(s3.METHODS))
   self.assertIn('cross_ablation',analysis)
   self.assertEqual(analysis['vs_random']['fixed']['n'],4)

 def test_estimator_required_for_estimator_methods(self):
  t=make_task()
  with self.assertRaises(ValueError):
   s3.assignments_for(t,'greedy',0,None)

if __name__=='__main__':unittest.main()
