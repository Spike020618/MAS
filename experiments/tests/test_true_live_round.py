import tempfile,unittest
from experiments.coordination_game_v1.live_runtime import DurableRuntime,UnknownRequest
from experiments.coordination_game_v1.schema import Action,MemberState
from experiments.coordination_game_v1.value import FrozenValues
from experiments.coordination_game_v1.true_live_round import run_live_round
class LiveRoundTests(unittest.TestCase):
 def test_real_message_path_and_no_run(self):
  calls=[]
  def transport(req):
   import json
   payload=json.loads(req['messages'][-1]['content']);calls.append(payload)
   return {'parsed':{'answer':'answer','observations':['fact'],'citation_ids':list(payload['sources'])}}
  with tempfile.TemporaryDirectory()as d:
   plan={'max_requests':3,'max_cny':1,'input_byte_cap':10000,'output_token_cap':768}
   runtime=DurableRuntime(d,plan,transport,False,1)
   task={'id':'q','question':'question','sources':[{'id':s,'title':s,'sentences':['fact']}for s in ('a','b')]}
   args=(runtime,task,[MemberState('i'),MemberState('j')],[],[Action('x','i','r',('a',)),Action('y','j','r',('b',))],FrozenValues({'x':1,'y':1}),'frozen-fixture')
   with self.assertRaises(RuntimeError):run_live_round(*args)
   self.assertEqual(calls,[])
   runtime.run=True;out=run_live_round(*args)
   self.assertEqual(len(calls),3);self.assertNotIn('observations',calls[0]);self.assertEqual(set(calls[2]['sources']),{'a','b'});self.assertEqual(len(out['executed_action_ids']),2)
 def test_unknown_review_blocks_readout(self):
  calls=[]
  def transport(req):calls.append(req);raise RuntimeError('unknown network completion')
  with tempfile.TemporaryDirectory()as d:
   runtime=DurableRuntime(d,{'max_requests':2,'max_cny':1,'input_byte_cap':10000,'output_token_cap':768},transport,True,1)
   args=(runtime,{'id':'q','question':'q','sources':[{'id':'a','title':'a','sentences':['fact']}]},[MemberState('i')],[],[Action('x','i','r',('a',))],FrozenValues({'x':1}),'fixture')
   with self.assertRaises(RuntimeError):run_live_round(*args)
   with self.assertRaises(UnknownRequest):run_live_round(*args)
   self.assertEqual(len(calls),1)
if __name__=='__main__':unittest.main()
