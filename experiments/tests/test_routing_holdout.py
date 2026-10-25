import unittest
from experiments.mechanism.validate_routing_complementarity import plans,holm,exact_pair
from experiments.mechanism.prepare_upgrade_p2 import native_mc


class HoldoutRoutingTest(unittest.TestCase):
    def test_label_blind_public_option_order(self):
        a=native_mc('test','id','question',['a','b','c'],0)
        b=native_mc('test','id','question',['a','b','c'],2)
        self.assertEqual(a['question'],b['question'])
        self.assertEqual(a['choices'],b['choices'])

    def test_union_and_matched_quotas(self):
        cases=[{'id':str(i),'question':'x'*i,'choices':['yes','no']} for i in range(1,6)]
        def report(a,y):return {'answer':a,'probabilities':{'yes':y,'no':1-y}}
        obs={c['id']:[report('yes',.95)]*3 for c in cases}
        obs['1']=[report('yes',.8),report('yes',.8),report('no',.2)]
        obs['2']=[report('yes',.51)]*3
        p=plans(cases,obs)
        self.assertEqual(p['native_quota'],1)
        self.assertEqual(p['matched_quota'],2)
        self.assertEqual(set(p['policies']['hybrid_union']),{'1','2'})
        self.assertTrue(all(len(ids)==2 for ids in p['policies'].values()))

    def test_holm_and_pair(self):
        self.assertEqual(holm({'a':.01,'b':.04}),{'a':.02,'b':.04})
        rows=[{'bounds':{'a':[0,0],'b':[1,1]}}]*2
        self.assertEqual(exact_pair(rows,'a','b')['net'],2)
        self.assertEqual(exact_pair(rows,'a','b')['exact_mcnemar_p'],.5)

    def test_no_probability_repair(self):
        from experiments.mechanism.audit_probability_compatibility import compatible_decode
        result=compatible_decode('{"answer":"yes","probabilities":{"yes":"0.8","no":"0.8"}}',['yes','no'])
        self.assertIsNone(result['probabilities'])
