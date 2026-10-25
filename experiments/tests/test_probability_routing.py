import json
import unittest
from experiments.mechanism.probe_probability_routing import decode,features,select,request,auc


class ProbabilityRoutingTest(unittest.TestCase):
    def test_distribution_validation(self):
        valid={'answer':'yes','probabilities':{'yes':.6,'no':.4}}
        self.assertIsNotNone(decode(json.dumps(valid),['yes','no'])['probabilities'])
        for distribution in [{'yes':.8,'no':.4},{'yes':True,'no':False},{'yes':.4,'no':.6},{'yes':1}]:
            valid['probabilities']=distribution
            self.assertIsNone(decode(json.dumps(valid),['yes','no'])['probabilities'])
        self.assertIsNone(decode('{"answer":"yes","answer":"no"}',['yes','no'])['answer'])

    def test_selection_only_uses_low_observations(self):
        cases=[{'id':str(i),'question':'q'*i,'choices':['yes','no']} for i in range(1,5)]
        def report(a,v):return {'answer':a,'probabilities':{'yes':v,'no':1-v}}
        observations={c['id']:[report('yes',.9)]*3 for c in cases}
        observations['1']=[report('yes',.6),report('no',.4),report('yes',.6)]
        result=select(cases,observations)
        self.assertEqual(result['quota_questions'],1)
        self.assertTrue(all(len(v)==1 for v in result['policies'].values()))
        self.assertEqual(result['policies']['probability'],['1'])
        self.assertEqual(result,select(cases,observations))

    def test_uncertainty_without_fake_normalization(self):
        r={'answer':'yes','probabilities':None}
        self.assertIsNone(features([r]*3,['yes','no'])['ensemble_uncertainty'])

    def test_same_prompt_only_backend_changes(self):
        case={'question':'test','choices':['yes','no']}
        a=request(case,0,'zhipu_flash');b=request(case,0,'deepseek')
        self.assertEqual(a['messages'],b['messages'])
        self.assertEqual(a['max_tokens'],b['max_tokens'])

    def test_auc(self):
        self.assertEqual(auc([.9,.1],[True,False]),1.)
        self.assertEqual(auc([.5,.5],[True,False]),.5)
        self.assertIsNone(auc([.5],[True]))
