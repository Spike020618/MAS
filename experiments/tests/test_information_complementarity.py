import copy
import unittest
from experiments.mechanism.study_information_complementarity import BACKENDS, request, majority, summarize


class ComplementarityTests(unittest.TestCase):
    def case(self):
        return {'id':'fixture:0','question':'Q','choices':['yes','no'],'split':'evaluation','label':'secret'}

    def test_requests_have_identical_information_and_no_labels(self):
        c=self.case()
        requests=[request(c,b,'fixed-model',i) for b in BACKENDS for i in range(3)]
        self.assertTrue(all(r['messages']==requests[0]['messages'] for r in requests))
        self.assertNotIn('secret',str(requests))
        self.assertTrue(all(r['max_tokens']==800 and r['temperature']==.7 for r in requests))
        self.assertTrue(all('seed' not in r for r in requests))

    def test_no_majority_abstains(self):
        self.assertEqual(majority(['yes','no','abstain']),'abstain')
        self.assertEqual(majority(['option_1','option_2','option_3']),'abstain')
        self.assertEqual(majority(['yes','yes','no']),'yes')

    def test_correct_minority_and_wrong_majority_are_separate(self):
        c=self.case(); slots={}
        for b in BACKENDS:
            for i in range(3):
                slots[b+'/'+str(i)]={'answer':'yes' if b=='ali_qwen' else 'no',
                     'tokens':10,'wall_seconds':1.,'valid':True}
        r=summarize([c],{c['id']:slots},{c['id']:'yes'})
        self.assertEqual(r['diagnostics']['heterogeneous0_any_correct'],1)
        self.assertEqual(r['metrics']['heterogeneous0']['correct'],0)
        self.assertEqual(r['diagnostics']['heterogeneous0_correct_available_but_vote_wrong'],1)
        self.assertEqual(r['diagnostics']['unique_backend0_correct']['ali_qwen'],1)
        self.assertEqual(r['metrics']['heterogeneous0']['tokens'],30)
        self.assertEqual(r['metrics']['single0/ali_qwen']['calls'],1)
        self.assertEqual(r['metrics']['homogeneous/ali_qwen']['calls'],3)

    def test_unknown_cost_remains_unknown(self):
        c=self.case()
        slots={b+'/'+str(i):{'answer':'yes','tokens':None,'wall_seconds':1.,'valid':True}
               for b in BACKENDS for i in range(3)}
        r=summarize([c],{c['id']:slots},{c['id']:'yes'})
        self.assertIsNone(r['metrics']['heterogeneous0']['tokens'])
        self.assertEqual(r['metrics']['heterogeneous_all27_expected']['correct'],1.)

    def test_all27_not_27_independent_trials(self):
        c=self.case()
        slots={b+'/'+str(i):{'answer':'yes' if i==0 else 'no','tokens':10,'wall_seconds':1.,'valid':True}
               for b in BACKENDS for i in range(3)}
        r=summarize([c],{c['id']:slots},{c['id']:'yes'})
        self.assertEqual(r['metrics']['heterogeneous_all27_expected']['n'],1)
        self.assertAlmostEqual(r['metrics']['heterogeneous_all27_expected']['correct'],7/27)


if __name__=='__main__':unittest.main()
