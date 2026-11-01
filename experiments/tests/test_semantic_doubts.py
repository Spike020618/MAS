import copy,unittest
from experiments.evidence_consensus.semantic_doubts import propose_doubts
class SemanticDoubtTests(unittest.TestCase):
 def test_specific_categories_and_body_independence(self):
  t={'kind':'rec','candidates':[str(i)for i in range(12)]};s={'decision':[str(i)for i in range(12)],'catalog':{'s':{'title':'premise','text':'SECRET'}}};r=[{'A':[{'id':'a','text':'The seeker prefers comedies'}],'I':[{'claim_id':'a','source_id':'s','relation':'support'},{'claim_id':'a','source_id':'s','relation':'refute'}],'decision':['11','2']},{'A':[{'id':'b','text':'An alternative film matches the request'}],'decision':['0']}]
  a=propose_doubts(t,s,r);s2=copy.deepcopy(s);s2['catalog']['s']['text']='OTHER BODY';self.assertEqual(a,propose_doubts(t,s2,r));self.assertEqual({p['kind']for p in a},{'premise','conflict','omission'});self.assertTrue(all(p['target_claim_ids']and p['question']for p in a))
 def test_foreign_claim_and_source_not_admitted(self):
  t={'kind':'qa'};s={'decision':'yes','catalog':{'s':{'title':'source','text':'body'}}};r=[{'A':[{'id':'a','text':'premise'}],'proposed_doubts':[{'kind':'premise','question':'x','target_claim_ids':['foreign'],'source_ids':['bad']}]}];a=propose_doubts(t,s,r);self.assertEqual(len(a),1);self.assertEqual(a[0]['source_ids'],['s'])
