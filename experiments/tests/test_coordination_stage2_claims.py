import json,math,unittest
from pathlib import Path
from experiments.coordination_game_v1 import stage2_claims as s2

class Stage2PrepareTests(unittest.TestCase):
 def test_claim_packet_partition_and_probes(self):
  row={'claim_id':'7','claim':'c','claim_label':'SUPPORTS',
       'evidences':[{'evidence_id':f'x{i}','article':f'A{i}','evidence':f'text {i}','entropy':0,'votes':[],'evidence_label':'SUPPORTS'}for i in range(6)]}
  packets,candidates,catalog=s2.claim_packet(row)
  self.assertEqual(len(catalog),6)
  for m in range(3):
   own=set(packets[str(m)])
   self.assertTrue(own)
   self.assertEqual(len(candidates[str(m)]),2)
   self.assertTrue(set(candidates[str(m)])-own==set(candidates[str(m)]))  # probes outside packet
  # every evidence appears in exactly one packet
  seen=[sid for m in range(3) for sid in packets[str(m)]]
  self.assertEqual(sorted(seen),[f'e{i}'for i in range(6)])

 def test_preregistration_gate_frozen(self):
  self.assertEqual(s2.PREREGISTRATION['train_claims'],120)
  self.assertEqual(s2.PREREGISTRATION['cal_claims'],40)
  self.assertEqual(s2.PREREGISTRATION['members'],3)
  self.assertGreaterEqual(s2.PREREGISTRATION['gate']['cal_auc_beneficial_vs_harmful_min'],0.5)

 def test_claim_message_deterministic_and_label_free(self):
  # dict key insertion order must not affect serialization (receipts persist sorted keys);
  # the observations list itself is byte-stable because it comes from the same stored response.
  a=s2.claim_message('claim text',{'e0':'src'},'revise',{'prior_verdict':'supported','prior_observations':['a','b']})
  b=s2.claim_message('claim text',{'e0':'src'},'revise',{'prior_observations':['a','b'],'prior_verdict':'supported'})
  self.assertEqual(a['messages'][1]['content'],b['messages'][1]['content'])
  self.assertEqual(a['max_tokens'],512)
  self.assertIn('verdict',a['messages'][0]['content'])
  self.assertNotIn('gold',json.dumps(a))

 def test_verdict_of(self):
  self.assertEqual(s2.verdict_of({'parsed':{'verdict':' supported ','observations':[]}}),('supported',[],True))
  self.assertEqual(s2.verdict_of({'parsed':{'verdict':'maybe'}}),(None,[],False))
  self.assertEqual(s2.verdict_of({'parsed':None}),(None,[],False))

 def test_action_features_label_free(self):
  task={'id':'claim:1','claim':'global warming polar bears','partition':'calibration',
        'packets':{'0':['e0'],'1':['e1'],'2':['e2']},'catalog':{'e0':'warming text','e1':'bears text','e2':'other'}}
  r={'claim_id':'claim:1','member':0,'action_index':0,'action_source':'e1','initial_verdict':'refuted'}
  f=s2.action_features(task,r)
  self.assertNotIn('gold',json.dumps(f))
  self.assertIn('prior_verdict:refuted',f)
  self.assertGreater(f['action_claim_coverage'],0)

class Stage2IntegrationTests(unittest.TestCase):
 def test_fake_transport_collect_seal_and_fit(self):
  import tempfile
  from experiments.coordination_game_v1.live_runtime import DurableRuntime
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp)
   # synthetic claim pool file
   pool=[{'claim_id':str(i),'claim':f'claim {i}','claim_label':['SUPPORTS','REFUTES','NOT_ENOUGH_INFO'][i%3],
          'evidences':[{'evidence_id':f'x{i}{j}','article':f'A{i}{j}','evidence':f'evidence text {i} {j}','entropy':0,'votes':[],'evidence_label':'SUPPORTS'}for j in range(6)]}
         for i in range(10)]
   s2.CLAIM_SOURCE=root/'pool.jsonl'
   s2.CLAIM_SOURCE.write_text('\n'.join(json.dumps(r)for r in pool))
   s2.exclusions=lambda:set()
   # deterministic pool order so train/cal partitions cover all three labels
   s2._hash_order=lambda items,key:list(items)
   self.assertEqual(s2.prepare(root,train_count=6,cal_count=4),{'tasks':10,'train':6,'cal':4,'model_api_calls':0})
   tasks=json.loads((root/'tasks.json').read_text())
   self.assertEqual(len(tasks),10)
   self.assertEqual({t['partition']for t in tasks},{'train','calibration'})
   vault=json.loads((root/'private/scoring.json').read_text())
   self.assertEqual(len(vault),10)
   # action-alternating verdicts covering +1/0/-1 outcomes
   calls={'n':0}
   class FakeT:
    def __call__(self,request):
     content=request['messages'][1]['content']
     if 'initial' in content:verdict='not_enough_info'
     else:
      verdict='supported' if calls['n']%2==0 else 'refuted';calls['n']+=1
     return {'http_status':200,'parsed':{'observations':['o'],'citation_ids':['e0'],'verdict':verdict},'usage':{}}
   s2.DurableRuntime=lambda out,plan,transport,run,approved_cny:DurableRuntime(out,plan,FakeT(),run=True,approved_cny=1.0)
   out=root/'collect'
   result=s2.collect(root,out,approved_cny=1.0)
   self.assertEqual(result['records'],10*3*2)
   self.assertEqual(result['budget_violations'],0)
   panel=json.loads((out/'panel.json').read_text())
   self.assertEqual(panel['seal']['expected_records'],60)
   self.assertIn('generation',panel)
   # labels never leaked into slots
   for p in (out/'slots').glob('*.json'):
    raw=p.read_text()
    self.assertNotIn('"gold"',raw)
   # outcomes: action verdicts alternate supported/refuted from an NEI initial
   report=s2.fit(out,root)
   self.assertEqual(report['rows'],60)
   self.assertEqual(report['cal_rows'],24)
   # gold=supported claims -> +1 value; refuted/NEI gold -> 0 (NEI revised verdict still supported!=NEI... value computed vs gold)
   self.assertIn('value_distribution',report)
   self.assertIsNotNone(report['direction_model'])

if __name__=='__main__':unittest.main()
