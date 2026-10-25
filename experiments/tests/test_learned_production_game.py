from fractions import Fraction
import unittest
from experiments.core.mechanism import learned_production_game as g


def slots(a='yes',b='no',c='yes',x='yes',y='yes'):
    return {k:{'answer':v,'tokens':100} for k,v in
            zip(('fast0','fast1','fast2','think1','think2'),(a,b,c,x,y))}


class ProductionTests(unittest.TestCase):
    def test_majority_and_abstention(self):
        self.assertEqual(g.majority(['yes','no','abstain']),'abstain')
        self.assertEqual(g.majority(['yes','yes','no']),'yes')

    def test_actions_change_real_report_path(self):
        s=slots('yes','no','no','yes','yes')
        self.assertEqual(g.outcome(s,(0,0)),'no')
        self.assertEqual(g.outcome(s,(1,0)),'yes')
        self.assertEqual(g.outcome(s,(0,1)),'yes')

    def test_exact_potential_identity_and_no_profitable_deviation(self):
        t=g.table([{'slots':slots('yes','no','no','yes','yes'),'label':'yes'}])
        for p in g.PROFILES:
            for i in range(2):
                q=list(p);q[i]=1-q[i]
                self.assertEqual(g.field(t,q,'utilities',i)-g.field(t,p,'utilities',i),
                                 g.field(t,q,'potential')-g.field(t,p,'potential'))
        r=g.best_response(t)
        self.assertEqual(r['profile'],[1,0])
        self.assertEqual(r['empirical_regrets'],['0','0'])

    def test_equilibrium_is_not_global_optimum(self):
        # Generic coordination trap is a unit fixture, not an LLM result.
        t={}
        for p in g.PROFILES:
            correct=Fraction(int(p==(1,1)));costs=[Fraction(a,10) for a in p]
            t[''.join(map(str,p))]={'potential':str(correct-sum(costs)),
                                  'utilities':[str(correct-c) for c in costs]}
        self.assertEqual(g.best_response(t)['profile'],[0,0])
        self.assertGreater(g.field(t,(1,1),'potential'),g.field(t,(0,0),'potential'))

    def test_frozen_policy_uses_no_labels_or_domain(self):
        s=slots(); state=g.context({k:s[k] for k in ('fast0','fast1','fast2')})
        policy=g.fit([{'slots':s,'label':'yes','context':state} for _ in range(3)])
        self.assertEqual(g.decide(policy,{k:s[k] for k in ('fast0','fast1','fast2')},'game'),[0,0])
        self.assertEqual(policy['contexts'][state]['n'],3)
        initial={k:{'answer':'yes'} for k in ('fast0','fast1','fast2')}
        self.assertEqual(g.decide(policy,initial,'game'),policy['contexts']['global']['game']['profile'])

    def test_invalid_action_and_small_context_fallback(self):
        with self.assertRaises(ValueError):g.outcome(slots(),(0,2))
        with self.assertRaises(ValueError):g.fit([])
        policy=g.fit([{'slots':slots(),'label':'yes','context':'disagreement'}])
        self.assertEqual(set(policy['contexts']),{'global'})


if __name__=='__main__':unittest.main()
