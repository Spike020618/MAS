import unittest
from experiments import state_union_action_value as u
from experiments.bound_opportunity_probe import fixtures
from experiments.calibrated_action_value import OutcomeModel
from experiments.consensus_mechanism import core as c
class SizeOutcome:
    def probs(self,X,prior=False):
        return [{1:.9 if x['source_count']==2 else .1,0:.1 if x['source_count']==2 else .9,-1:0.}for x in X]
class UnionTests(unittest.TestCase):
    def fixture(self):
        row=fixtures()[1];row['task']['metadata']['given_claim']='Record VX17 is approved.'
        bundle={'outcome':SizeOutcome(),'validity':OutcomeModel().fit([{}],[1]),'cost_weight':.1}
        return row,bundle
    def test_second_action_counts_known_complement(self):
        row,bundle=self.fixture();origin=c.fresh(row['task']);state=c.fresh(row['task']);c.expose(row['task'],state,['A'])
        values=u.conditional_values(row['task'],state,origin,bundle)
        self.assertEqual(len(values),1);self.assertEqual(values[0]['source_ids'],['B'])
        self.assertAlmostEqual(values[0]['set_value_before'],.1)
        self.assertAlmostEqual(values[0]['set_value_after'],.9)
        self.assertAlmostEqual(values[0]['gains'][0],.8)
    def test_initial_joint_value_and_constraints(self):
        row,bundle=self.fixture();state=c.fresh(row['task'])
        chosen,trace=u.select(row['task'],state,state,bundle,'game',1,2)
        self.assertEqual(len(chosen),1);self.assertEqual(set(chosen[0]['source_ids']),{'A','B'})
        self.assertAlmostEqual(chosen[0]['gains'][0],.9)
        self.assertLessEqual(trace['allocation']['max_feasible_deviation'],1e-10)
if __name__=='__main__':unittest.main()
