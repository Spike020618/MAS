import unittest
from experiments.coordination_game_v1.schema import Action,Review
from experiments.coordination_game_v1.value import FrozenValues
from experiments.coordination_game_v1.ledger import Ledger
from experiments.coordination_game_v1.true_policy_adapter import plan_round,execute_round,MODES

class TruePolicyTests(unittest.TestCase):
 def setUp(self):
  self.actions=(Action('a','i','r',('A',)),Action('b','j','r',('B',)))
 def test_complementarity_and_mutual_pair_move(self):
  value=FrozenValues({'a':-1,'b':-1},{('a','b'):3})
  stop,unary=plan_round({},self.actions,value,3,'game')
  pair,cert=plan_round({},self.actions,value,3,'pair_game')
  self.assertEqual(stop.assignments,());self.assertEqual(len(pair.assignments),2)
  self.assertEqual(cert['permitted_pair_residual'],0)
  self.assertTrue(all(v>=0 for v in cert['steps'][0]['member_gains'].values()))
 def test_negative_values_stop(self):
  for mode in MODES:
   plan,_=plan_round({},self.actions,FrozenValues({'a':-1,'b':-2}),3,mode)
   self.assertEqual(plan.assignments,())
 def test_uniform_domain_and_central_bound(self):
  actions=self.actions+(Action('c','i','r',('C',)),)
  value=FrozenValues({'a':1,'b':2,'c':3})
  for mode in MODES:
   plan,cert=plan_round({},actions,value,3,mode)
   self.assertEqual(len({a.member_id for a in plan.assignments}),len(plan.assignments))
   self.assertGreaterEqual(cert['central_gap'],-1e-10)
   self.assertEqual(len(cert['source_hashes']['solver.py']),64)
 def test_execution_covers_whole_frozen_configuration(self):
  seen=[]
  def review(action,state):seen.append(action.action_id);return Review(action.action_id,(action.action_id,),action.source_ids)
  out=execute_round({'version':1},self.actions,FrozenValues({'a':1,'b':2}),3,Ledger(3,20000),review,lambda state,evidence:len(evidence),'game')
  self.assertEqual(len(seen),2);self.assertEqual(out['result']['answer'],2)
 def test_no_cbba_alias(self):
  with self.assertRaises(ValueError):plan_round({},self.actions,FrozenValues({'a':1,'b':2}),3,'cbba')
if __name__=='__main__':unittest.main()
