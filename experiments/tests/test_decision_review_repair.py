import unittest
from experiments.consensus_mechanism.account_guard import check,AccountUnavailable
from experiments.consensus_mechanism import decision_review as d
class RepairTest(unittest.TestCase):
 def test_account_refusal_is_fatal_not_event_fallback(self):
  with self.assertRaises(AccountUnavailable):check({'status_code':400,'body':'Access denied, account good standing overdue-payment'})
  self.assertFalse(issubclass(AccountUnavailable,RuntimeError))
 def test_other_bad_request_not_misclassified(self):check({'status_code':400,'body':'invalid JSON parameter'})
 def test_success(self):check({'status_code':200,'body':'{}'})
if __name__=='__main__':unittest.main()

class DecisionPathTest(unittest.TestCase):
 def fixture(self,kind):
  import json,copy
  from pathlib import Path
  from experiments.consensus_mechanism import core as c
  tasks=json.loads(Path('experiments/results/consensus_mechanism_paper_v2/data/train.json').read_text());t=copy.deepcopy(next(t for t in tasks if t['kind']==kind));t['metadata']['decision_linked_v1']=True;s=c.fresh(t)
  if kind=='rec':s['decision']=c.candidate_ids(t)
  sids=[k for k in t['source_catalog']][:2]
  if len(sids)<2:raise ValueError('fixture needs two catalog sources')
  c.expose(t,s,sids);s['claims']['c']={'id':'c','text':'A task premise','active':True}
  spans=c.spans(t);span=next(k for k,v in spans.items()if v['source_id']==sids[0]);other=next(k for k,v in spans.items()if v['source_id']==sids[1])
  return t,s,{'target_id':'c','source_ids':[sids[0]]},span,{'source_id':sids[1],'span_id':other}
 def test_qa_patch_reaches_state(self):
  from experiments.consensus_mechanism import core as c
  t,s,a,span,other=self.fixture('qa');v=next(v for v in t['options']if v!=s['decision']);report={'target_id':'c','verdict':'refute','span_ids':[span],'rationale':'Evidence contradicts the premise','patch':{'operation':'set_answer','value':v},'decision_effect':{'choice':'revise','reason':'The answer-critical premise changes','candidate_pair':None}}
  class API:
   def call(self,slot,req):return {'report':report,'usage':{},'model':'fixture'}
  result,event=c.invoke(API(),'test',t,s,'review',1,action=a,reviewer=0);self.assertTrue(event['valid']);self.assertTrue(event['accepted']);self.assertEqual(result['decision'],v);self.assertFalse(result['claims']['c']['active']);self.assertEqual(event['revision_basis'],'assigned')
 def test_recommendation_swap_and_preservation(self):
  from experiments.consensus_mechanism import core as c
  t,s,a,span,other=self.fixture('rec');inside,outside=s['decision'][1],s['decision'][10];r={'target_id':'c','verdict':'refute','span_ids':[span],'rationale':'Comparative evidence','patch':{'operation':'swap_candidate','a':inside,'b':outside},'decision_effect':{'choice':'revise','reason':'Outside candidate better matches the current request','candidate_pair':[inside,outside]}};refs,basis=d.validate(t,s,r,1,a);self.assertEqual(basis,'assigned');after,accepted=c.commit(t,s,r,'review',1,refs);self.assertTrue(accepted);self.assertEqual(after['decision'][1],outside);self.assertEqual(after['decision'][0],s['decision'][0]);self.assertEqual(after['decision'][2:10],s['decision'][2:10])
 def test_keep_cannot_smuggle_patch(self):
  t,s,a,span,other=self.fixture('qa');r={'target_id':'c','verdict':'support','span_ids':[span],'rationale':'x','patch':{'operation':'set_answer','value':t['options'][0]},'decision_effect':{'choice':'keep','reason':'x'}}
  with self.assertRaises(ValueError):d.validate(t,s,r,1,a)
 def test_new_evidence_requirement_not_relaxed(self):
  t,s,a,span,other=self.fixture('qa');r={'target_id':'c','verdict':'refute','span_ids':[],'rationale':'x','patch':{'operation':'set_answer','value':t['options'][0]},'decision_effect':{'choice':'revise','reason':'x'}}
  with self.assertRaises(ValueError):d.validate(t,s,r,1,a)
 def test_initial_basis_revision_accepted_and_classified(self):
  t,s,a,span,other=self.fixture('qa');v=next(v for v in t['options']if v!=s['decision'])
  r={'target_id':'c','verdict':'refute','span_ids':[other['span_id']],'rationale':'Re-reading the initial evidence justifies the correction','patch':{'operation':'set_answer','value':v},'decision_effect':{'choice':'revise','reason':'Initial evidence re-read','candidate_pair':None}}
  refs,basis=d.validate(t,s,r,1,a);self.assertEqual(basis,'initial');self.assertTrue(refs)
 def test_linked_invoke_accepts_initial_basis_patch_end_to_end(self):
  from experiments.consensus_mechanism import core as c
  t,s,a,span,other=self.fixture('qa');v=next(v for v in t['options']if v!=s['decision'])
  report={'target_id':'c','verdict':'refute','span_ids':[other['span_id']],'rationale':'Re-reading initial evidence justifies the correction','patch':{'operation':'set_answer','value':v},'decision_effect':{'choice':'revise','reason':'Initial evidence re-read','candidate_pair':None}}
  class API:
   def call(self,slot,req):return {'report':report,'usage':{},'model':'fixture'}
  result,event=c.invoke(API(),'test',t,s,'review',1,action=a,reviewer=0)
  self.assertTrue(event['valid']);self.assertEqual(event['revision_basis'],'initial');self.assertTrue(event['accepted']);self.assertEqual(result['decision'],v)
 def test_strict_contract_still_enforced_for_non_linked(self):
  from experiments.consensus_mechanism import core as c
  t,s,a,span,other=self.fixture('qa');v=next(v for v in t['options']if v!=s['decision'])
  r={'target_id':'c','verdict':'refute','span_ids':[other['span_id']],'rationale':'x','patch':{'operation':'set_answer','value':v}}
  with self.assertRaises(ValueError):c.validate_report(t,s,r,'review',1,a)
 def test_citation_policy_texts(self):
  from experiments.consensus_mechanism import core as c
  t,s,a,span,other=self.fixture('qa')
  plain=c.request(t,s,'review',1,(),a);linked=d.request(t,s,1,(),a)
  self.assertIn('assigned_action.source_ids',plain['messages'][0]['content']);self.assertNotIn('initial-basis',plain['messages'][0]['content'])
  self.assertIn('initial-basis',linked['messages'][0]['content'])

if __name__=='__main__':unittest.main()
