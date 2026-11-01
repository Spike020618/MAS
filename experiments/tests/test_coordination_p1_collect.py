import json,unittest
from pathlib import Path
from experiments.coordination_game_v1.collect import branch_kind,matched,readout_answer,task_sources,source_text,build_plan
from experiments.coordination_game_v1.collection import branch_schedule,message

class AdapterTests(unittest.TestCase):
 def test_branch_kinds_and_matching(self):
  self.assertEqual(branch_kind('empty'),'empty');self.assertEqual(branch_kind('single_2_1'),'single')
  self.assertEqual(branch_kind('joint_1'),'same');self.assertEqual(branch_kind('cross_0'),'cross')
  self.assertEqual(matched('joint_1'),{'matched_single_branches':['single_1_0','single_1_1']})
  self.assertEqual(matched('cross_2'),{'matched_single_branches':['single_2_0','single_0_1'],'matched_same_branch':'joint_2'})
  self.assertEqual(matched('single_0_0'),{});self.assertEqual(matched('empty'),{})
 def test_readout_answer_validity(self):
  self.assertEqual(readout_answer({'parsed':{'answer':' Yes '}}),('Yes',True))
  self.assertEqual(readout_answer({'parsed':{'answer':''}}),('noanswer',False))
  self.assertEqual(readout_answer({'parsed':None}),('noanswer',False))
  self.assertEqual(readout_answer({'parsed':{'observations':[]}}),('noanswer',False))
 def test_task_sources_structure(self):
  task={'id':'t1','source_order':['s0','s1','s2','s3','s4','s5'],
   'member_initial_sources':{'0':['s0'],'1':['s1'],'2':['s2']},
   'sources':[{'id':f's{i}','title':f'T{i}','sentences':[f'sent {i}']}for i in range(6)]}
  initial,action=task_sources(task)
  self.assertEqual(len(initial),3);self.assertEqual(list(initial[0]),['s0'])
  self.assertEqual([list(a)[0]for a in action],['s3','s4','s5'])
  self.assertIn('T0: sent 0',source_text(task['sources'][0]))
 def test_branch_schedule_shape(self):
  schedule=branch_schedule()
  self.assertEqual(len(schedule),16)
  kinds=[branch_kind(name)for name,_ in schedule]
  from collections import Counter
  self.assertEqual(Counter(kinds),{'empty':1,'single':9,'same':3,'cross':3})
  for _,assignments in schedule:
   for m,indices in assignments:self.assertTrue(all(0<=s<3 for s in indices))
 def test_message_has_no_shared_answer(self):
  req=message('q',{'s0':'text'},'review',[{'observations':['x']}])
  self.assertNotIn('answer',json.dumps(req['messages'][1]['content']))
  self.assertEqual(req['max_tokens'],768)
 def test_build_plan_freezes_budget(self):
  manifest={'tasks':12,'repeats':2,'task_ids':[f'q{i}'for i in range(12)],'tasks_sha256':'x','source_split':'public_validation'}
  from experiments.coordination_game_v1.panel import budget as make_budget
  plan=build_plan(manifest,make_budget())
  self.assertEqual(plan['max_requests'],888);self.assertAlmostEqual(plan['max_cny'],1.5685632,places=6)

class CollectIntegrationTests(unittest.TestCase):
 def test_fake_transport_collect_and_records(self):
  import tempfile
  from experiments.coordination_game_v1.collect import collect,run_analysis
  from experiments.coordination_game_v1.live_runtime import DurableRuntime
  import experiments.coordination_game_v1.collect as collect_mod
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp);data=root/'data';data.mkdir()
   task={'id':'q1','question':'Is A the same as B?','source_order':['s0','s1','s2','s3','s4','s5'],
    'member_initial_sources':{'0':['s0'],'1':['s1'],'2':['s2']},'repeats':[0,1],
    'sources':[{'id':f's{i}','title':f'T{i}','sentences':[f'sent {i}']}for i in range(6)]}
   (data/'tasks.json').write_text(json.dumps([task]))
   (data/'manifest.json').write_text(json.dumps({'tasks':1,'repeats':2,'task_ids':['q1'],'tasks_sha256':'h','source_split':'public_validation'}))
   (data/'private').mkdir();(data/'private/scoring.json').write_text(json.dumps({'q1':{'answer':'yes'}}))
   class FakeTransport:
    def __call__(self,request):
     return {'http_status':200,'parsed':{'observations':['obs'],'citation_ids':['s0'],'answer':'yes'},'usage':{}}
   collect_mod.DurableRuntime=lambda out,plan,transport,run,approved_cny:DurableRuntime(out,plan,FakeTransport(),run=True,approved_cny=1.0)
   out=root/'out'
   result=collect(data,out,approved_cny=1.0)
   self.assertEqual(result['records'],32)  # 1 task x 2 repeats x 16 branches
   panel=json.loads((out/'panel.json').read_text())
   self.assertEqual(panel['seal']['expected_records'],384)
   self.assertEqual(panel['seal']['records_hash'],__import__('experiments.coordination_game_v1.schema',fromlist=['digest']).digest(panel['records']))
   analysis=run_analysis(out,data)
   self.assertEqual(analysis['status'],'incomplete')  # 32 != 384 expected by frozen protocol

if __name__=='__main__':unittest.main()

class ReplayStabilityTests(unittest.TestCase):
 def test_readout_request_identical_live_vs_stored_order(self):
  from experiments.coordination_game_v1.schema import digest
  live={'observations':['x'],'answer':'a','citation_ids':['s0']}
  stored={k:live[k]for k in sorted(live)}
  r1=message('q',{'s0':'t'},'readout',[live]);r2=message('q',{'s0':'t'},'readout',[stored])
  self.assertEqual(digest(r1),digest(r2))
  self.assertEqual(r1['messages'][1]['content'],r2['messages'][1]['content'])
