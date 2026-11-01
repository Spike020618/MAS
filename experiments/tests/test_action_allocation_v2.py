import unittest,copy
from experiments.evidence_consensus.action_allocation_v2 import ActionAllocation
class AllocationTests(unittest.TestCase):
 def actions(self):return [{'action_id':str(i),'source_unit':str(i),'doubt_id':'d','target_claim_ids':['c'],'net_gain':i+1.,'cost':.1}for i in range(4)]
 def test_potential_identity_all_deviations(self):
  g=ActionAllocation(self.actions(),reviewers=2,max_actions=2)
  for s in g.states():
   for i in range(2):
    for t in g.deviations(s,i):self.assertAlmostEqual(g.utility(t,i)-g.utility(s,i),g.potential(t)-g.potential(s))
 def test_fixed_independent_of_scores(self):
  a=self.actions();x=ActionAllocation(a).solve()['fixed']
  for r in a:r['net_gain']=-100
  self.assertEqual(x,ActionAllocation(a).solve()['fixed'])
 def test_duplicate_and_constraints(self):
  a=self.actions();a[1]['source_unit']=a[0]['source_unit'];g=ActionAllocation(a,max_actions=1,budget=.1)
  r=g.solve();self.assertTrue(g.feasible(r['game']));self.assertLessEqual(r['max_feasible_deviation'],1e-10);self.assertGreaterEqual(r['proxy_gap'],0)
  g=ActionAllocation(a,cost_weight=0);self.assertEqual(g.potential((0,1,None)),2.)
 def test_homogeneous_means_identical_gains(self):
  g=ActionAllocation(self.actions());self.assertEqual(g.gain(0,2),g.gain(2,2))
