import json,unittest
from pathlib import Path
from experiments.coordination_game_v1 import stage4_hetero as s4

class RoutingTests(unittest.TestCase):
 def test_member_of_all_patterns(self):
  self.assertEqual(s4.member_of('q1/repeat-0/initial-2'),2)
  self.assertEqual(s4.member_of('q1/repeat-1/single_1_2/check-0'),1)
  self.assertEqual(s4.member_of('q1/repeat-0/joint_0/check-0'),0)
  self.assertEqual(s4.member_of('q1/repeat-0/cross_2/check-0'),2)
  self.assertEqual(s4.member_of('q1/repeat-0/cross_2/check-1'),0)  # (2+1)%3
  self.assertEqual(s4.member_of('q1/repeat-0/joint_1/readout'),'reader')
  with self.assertRaises(ValueError):s4.member_of('q1/repeat-0/unknown')

 def test_preregistration_frozen(self):
  self.assertEqual(s4.MEMBER_PROVIDERS,('qwen_flash','glm_flash','qwen_flash'))
  self.assertEqual(s4.PREREGISTRATION['branches_per_repeat'],16)
  self.assertEqual(s4.PREREGISTRATION['engineering']['maximum_leakage_events'],0)

class HeteroIntegrationTests(unittest.TestCase):
 def test_fake_transport_collect_and_analyze(self):
  import tempfile
  from experiments.coordination_game_v1.live_runtime import DurableRuntime
  import experiments.coordination_game_v1.transport as transport_mod
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp);data=root/'data';data.mkdir()
   task={'id':'q1','question':'Is A the same as B?','source_order':['s0','s1','s2','s3','s4','s5'],
    'member_initial_sources':{'0':['s0'],'1':['s1'],'2':['s2']},'repeats':[0],
    'sources':[{'id':f's{i}','title':f'T{i}','sentences':[f'sent {i}']}for i in range(6)]}
   (data/'tasks.json').write_text(json.dumps([task]))
   (data/'manifest.json').write_text(json.dumps({'tasks':1,'repeats':1,'task_ids':['q1'],'tasks_sha256':'h','source_split':'public_validation'}))
   (data/'private').mkdir();(data/'private/scoring.json').write_text(json.dumps({'q1':{'answer':'yes'}}))
   calls={'n':0}
   class FakeT:
    def __init__(self,provider):self.provider=provider
    def __call__(self,request):
     calls['n']+=1
     content=request['messages'][1]['content']
     if '"readout"' in content:
      return {'http_status':200,'parsed':{'observations':['o'],'citation_ids':['s0'],'answer':'yes'},'usage':{}}
     return {'http_status':200,'parsed':{'observations':['obs'],'citation_ids':['s0'],'answer':'maybe'},'usage':{}}
   transport_mod.Transport=FakeT
   import experiments.coordination_game_v1.collect as collect_mod
   collect_mod.PacedTransport=lambda transport:transport  # bypass the shared live throttle ledger
   out=root/'out'
   # repeats field of manifest says 1; collect uses task['repeats']=[0] -> 16 records
   result=s4.collect(data,out,approved_total=1.0)
   self.assertEqual(result['records'],16)
   self.assertEqual(result['budget_violations'],0)
   panel=json.loads((out/'panel.json').read_text())
   self.assertEqual(panel['seal']['expected_records'],384)
   # every member sub-root sealed
   for sub in ('member-0','member-1','member-2','reader'):
    self.assertTrue((out/sub/'generation_complete.json').exists(),sub)
   # analysis via the frozen P1 protocol (expects 384 records -> reports incomplete status)
   from experiments.coordination_game_v1.collect import run_analysis
   analysis=run_analysis(out,data)
   self.assertIn(analysis['status'],('incomplete','sealed'))
   # labels never in any slot
   for sub in out.iterdir():
    if (sub/'slots').is_dir():
     for p in (sub/'slots').glob('*.json'):
      self.assertNotIn('"gold"',p.read_text())

if __name__=='__main__':unittest.main()
