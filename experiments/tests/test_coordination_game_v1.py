import unittest
from experiments.coordination_game_v1.schema import *
from experiments.coordination_game_v1.data import project,partition,question_key
from experiments.coordination_game_v1.state import visible
from experiments.coordination_game_v1.tasks import propose
from experiments.coordination_game_v1.actions import candidates
from experiments.coordination_game_v1.reviewer import request,validate
from experiments.coordination_game_v1.features import extract
from experiments.coordination_game_v1.value import FrozenValues
from experiments.coordination_game_v1.solver import solve,configurations
from experiments.coordination_game_v1.utility import utility
from experiments.coordination_game_v1.ledger import Ledger
from experiments.coordination_game_v1.runner import execute
from experiments.coordination_game_v1.panel import budget,branch_identity
from experiments.coordination_game_v1.report import counterexamples
from experiments.coordination_game_v1.evaluate import answer_score

class CoordinationP0(unittest.TestCase):
 def setUp(self):
  self.a=Action('a','i','r',('A',));self.b=Action('b','j','r',('B',));self.state={'public':['s'],'relation':'r'}
 def test_member_state_immutable(self):
  m=MemberState('i',('A',))
  with self.assertRaises(Exception):m.member_id='j'
 def test_public_projection_excludes_gold(self):
  row={'_id':'q','question':'Who?','context':[['Title',['text']]],'answer':'secret','supporting_facts':[['Title',0]]}
  p=project(row);self.assertNotIn('answer',p);self.assertNotIn('supporting_facts',p);self.assertNotIn('secret',str(p))
 def test_exposure_normalized_duplicate(self):
  row={'_id':'q','question':'Who?','context':[]}
  result=partition([row,{**row,'_id':'other','question':'WHO !'}]);self.assertEqual(len(result['eligible']),1)
  self.assertEqual(len(partition([row],exposed_questions=[question_key('Who?')])['eligible']),0)
 def test_request_no_peer_rationale(self):
  packet=request('q',MemberState('i',('A',),('secret-peer-conclusion',)),(),self.a,{'A':'text'},'r',0)
  self.assertNotIn('secret-peer-conclusion',str(packet));self.assertEqual(packet['sources'],{'A':'text'})
 def test_visibility_is_member_specific(self):
  m=MemberState('i',('C',));self.assertEqual(set(visible(m,(),self.a,{'A':'a','B':'b','C':'c'})),{'A','C'})
  with self.assertRaises(ValueError):visible(MemberState('j'),(),self.a,{'A':'a'})
 def test_disagreement_and_no_forced_tasks(self):
  self.assertEqual(propose({'r':'relation'},[{'r':'yes'},{'r':'yes'}]),[])
  self.assertEqual(propose({'r':'relation'},[{'r':'yes'},{'r':'no'}])[0].trigger,'disagreement')
 def test_equal_access_distinct_features(self):
  members=[MemberState('i',('A',)),MemberState('j',())];tasks=propose({'r':'r'},[{},{}]);actions=candidates(members,tasks,['A'])
  self.assertEqual(len(actions),2)
  f=[extract(m,a,{'A':'r'},'r')for m,a in zip(members,actions)]
  self.assertNotEqual(f[0]['already_read_fraction'],f[1]['already_read_fraction']);self.assertNotIn('member_id',f[0])
 def test_invisible_citation_invalid(self):
  r=validate(self.a,{'observations':['x'],'citation_ids':['B']},{'sources':{'A':'a'}});self.assertFalse(r.valid)
 def test_potential_identity(self):
  value=FrozenValues({'a':-.2,'b':.3},{('a','b'):1})
  old=(self.b,);new=(self.a,self.b)
  self.assertAlmostEqual(utility(value,new,'i')-utility(value,old,'i'),value(new)-value(old))
 def test_joint_can_escape_unilateral_stop(self):
  v=FrozenValues({'a':-1,'b':-1},{('a','b'):3})
  unary=solve((self.a,self.b),v,3);pair=solve((self.a,self.b),v,3,pair=True)
  self.assertEqual(unary['selected'],());self.assertEqual(pair['value'],1)
  self.assertTrue(all(all(g>=0 for g in s['member_gains'].values())for s in pair['steps']))
 def test_central_upper_bound(self):
  for x in (-2,0,1):
   for y in (-1,.5,3):
    v=FrozenValues({'a':x,'b':y},{('a','b'):2})
    for mode in ('game','block','greedy'):
     result=solve((self.a,self.b),v,3,mode,pair=True);self.assertLessEqual(result['value'],result['central_value']+1e-10)
 def test_strict_progress_and_residual(self):
  v=FrozenValues({'a':1,'b':2});r=solve((self.a,self.b),v,3)
  self.assertTrue(all(s['potential_gain']>0 for s in r['steps']));self.assertEqual(r['unilateral_residual'],0)
 def test_readout_in_budget(self):
  self.assertTrue(all(sum(a.calls for a in c)<=1 for c in configurations((self.a,self.b),2)))
  with self.assertRaises(ValueError):configurations((self.a,),0)
 def test_full_execution_one_snapshot(self):
  seen=[]
  def review(a,state):seen.append((a.action_id,digest(state)));state['public'].append('mutated');return Review(a.action_id,(a.action_id,),a.source_ids)
  result=execute(RoundPlan(digest(self.state),(self.a,self.b),1),self.state,Ledger(3,20000),review,lambda s,e:len(e))
  self.assertEqual(result['executed_action_ids'],['a','b']);self.assertEqual(result['answer'],2);self.assertEqual(seen[0][1],seen[1][1]);self.assertEqual(self.state['public'],['s'])
 def test_atomic_reservation_no_partial_dispatch(self):
  seen=[];ledger=Ledger(2,20000)
  with self.assertRaises(ValueError):execute(RoundPlan(digest(self.state),(self.a,self.b),1),self.state,ledger,lambda a,s:seen.append(a),lambda s,e:None)
  self.assertEqual(seen,[]);self.assertEqual(ledger.reserved_calls,0)
 def test_failure_does_not_hide_other_action(self):
  def review(a,s):
   if a==self.a:raise TimeoutError()
   return Review(a.action_id,('b',),a.source_ids)
  r=execute(RoundPlan(digest(self.state),(self.a,self.b),1),self.state,Ledger(3,20000),review,lambda s,e:len(e))
  self.assertEqual(r['answer'],1);self.assertEqual(len(r['failures']),1)
 def test_stale_state_rejected(self):
  with self.assertRaises(ValueError):execute(RoundPlan('stale',(self.a,),1),{},Ledger(2,20000),None,None)
 def test_repeat_identity_not_cached_as_repeat(self):
  self.assertNotEqual(branch_identity('q',0,'A',{}),branch_identity('q',1,'A',{}))
 def test_budget_888(self):
  b=budget();self.assertEqual(b['requests'],888);self.assertEqual(b['input_token_cap'],3637248);self.assertAlmostEqual(b['estimated_upper_cny_configured_rates'],1.5685632)
 def test_exchangeability_not_allocator_equivalence(self):
  r=counterexamples()['exchangeable_members'];self.assertLess(r['fixed_value'],r['game_value'])
 def test_native_answer_scoring(self):
  self.assertEqual(answer_score('The Eiffel Tower','Eiffel Tower')['f1'],1);self.assertEqual(answer_score('yes','no')['f1'],0)
if __name__=='__main__':unittest.main()
