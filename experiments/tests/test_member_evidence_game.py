import unittest
import numpy as np
from experiments.member_evidence_game import EvidenceGame
class GameTests(unittest.TestCase):
    def channel(self,i,ids):
        reliability=.7 if i==0 and ids==['A']else .95 if i==1 and ids==['B']else .5
        return np.array([[reliability,1-reliability,0,0],[1-reliability,reliability,0,0],[0,0,1,0]])
    def game(self):return EvidenceGame([.5,.5,0],['A','B'],self.channel,lambda i,a:.001,calls=2,sources=1)
    def test_resource_locked_game_has_individually_rational_exchange(self):
        unary=self.game().solve('game');exchange=self.game().solve('exchange');central=self.game().solve('central')
        self.assertLess(unary['potential'],exchange['potential'])
        self.assertAlmostEqual(exchange['potential'],central['potential'])
        self.assertAlmostEqual(sum(exchange['credits']),0.)
        self.assertLessEqual(exchange['max_unilateral_deviation'],1e-10)
        self.assertLessEqual(exchange['max_permitted_pair_deviation'],1e-10)
        pairs=[x for x in exchange['trace']if x['kind']=='pair_exchange'];self.assertTrue(pairs)
        for row in pairs:
            self.assertGreater(row['potential_gain'],0)
            self.assertTrue(all(g>0 for g in row['individual_net_gains'].values()))
            self.assertAlmostEqual(sum(row['transfers'].values()),0.)
    def test_uninformative_checks_are_not_forced(self):
        q=np.array([[.5,.5,0,0]]*3)
        game=EvidenceGame([.5,.3,.2],['A'],lambda i,a:q,lambda i,a:.001,sources=1)
        result=game.solve('exchange')
        self.assertEqual(result['selected'],[])
    def test_budget_and_no_duplicate_material(self):
        game=self.game()
        for mode in ['game','exchange','greedy','central']:
            result=game.solve(mode);ids=[s for a in result['selected']for s in a['source_ids']]
            self.assertLessEqual(len(result['selected']),2);self.assertLessEqual(len(ids),1)
            self.assertEqual(len(ids),len(set(ids)))
    def test_perfect_channel_value(self):
        q=np.concatenate([np.eye(3),np.zeros((3,1))],axis=1)
        game=EvidenceGame([.5,.3,.2],['A'],lambda i,a:q,lambda i,a:0.,calls=1,sources=1)
        self.assertAlmostEqual(game.solve('central')['potential'],.5)
if __name__=='__main__':unittest.main()

class TwoSourceLockTests(unittest.TestCase):
    def test_exchange_unlocks_disjoint_pairs_under_real_two_source_budget(self):
        def channel(i,ids):
            r=.7 if i==0 and ids==['A','B']else .95 if i==1 and ids==['C','D']else .5
            return np.array([[r,1-r,0,0],[1-r,r,0,0],[0,0,1,0]])
        def make():return EvidenceGame([.5,.5,0],['A','B','C','D'],channel,lambda i,a:.001,calls=2,sources=2)
        unilateral=make().solve('game');exchange=make().solve('exchange');central=make().solve('central')
        self.assertGreater(exchange['potential'],unilateral['potential'])
        self.assertAlmostEqual(exchange['potential'],central['potential'])
        self.assertEqual(exchange['selected'][0]['member'],1)
