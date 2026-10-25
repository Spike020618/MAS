import unittest
from itertools import combinations

from experiments.mechanism.probe_resource_allocation import (
    prepare, report_request, choice_request, body, random_profile_weight, fixed_profiles, PROFILES,
)


class AllocationProbeTest(unittest.TestCase):
    def test_input_only_sampling(self):
        cases=prepare()
        self.assertEqual(cases,prepare())
        self.assertEqual(len(cases),24)
        self.assertEqual(len({c['id'] for c in cases}),24)
        for domain in ['strategyqa','mmlu','truthfulqa']:
            self.assertEqual(sum(c['id'].startswith(domain+':') for c in cases),8)
        self.assertTrue(all('label' not in c for c in cases))

    def test_only_execution_model_changes(self):
        case=prepare()[0]
        a=report_request(case,0,'zhipu_flash')
        b=report_request(case,0,'deepseek')
        self.assertEqual(a['messages'],b['messages'])
        for field in ['max_tokens','temperature','response_format']:
            self.assertEqual(a[field],b[field])
        self.assertEqual(a['max_tokens'],600)

    def test_hypergeometric_matches_enumeration(self):
        for quota in range(7):
            allocations=list(combinations(range(6),quota))
            self.assertAlmostEqual(sum(random_profile_weight(6,quota,p) for p in PROFILES),1)
            for p in PROFILES:
                expected=sum(tuple(int(i in a) for i in range(3))==p for a in allocations)/len(allocations)
                self.assertAlmostEqual(random_profile_weight(6,quota,p),expected)

    def test_fixed_exact_quota(self):
        cases=prepare()
        for quota in range(73):
            result=fixed_profiles(cases,quota)
            self.assertEqual(sum(sum(p) for p in result.values()),quota)
            self.assertTrue(all(p in PROFILES for p in result.values()))

    def test_selector_json_mode_instruction(self):
        request=choice_request(prepare()[0],0)
        self.assertIn('json',' '.join(m['content'] for m in request['messages']).lower())
        self.assertNotIn('label',request)

    def test_transport_failures_not_scored(self):
        import json
        with self.assertRaises(ValueError):body({'response':{'http_status':400}})
        value={'model':'example','usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2},
               'choices':[{'finish_reason':'length'}]}
        with self.assertRaises(ValueError):
            body({'response':{'http_status':200,'body':json.dumps(value)}})

    def test_paired_report_counts(self):
        from experiments.mechanism.report_resource_allocation import paired
        rows=[{'a':0,'b':1},{'a':0,'b':1},{'a':1,'b':0},{'a':1,'b':1}]
        self.assertEqual(paired(rows,'a','b'),(2,1,1.))
        self.assertEqual(paired([{'a':1,'b':1}],'a','b'),(0,0,1.))

    def test_missing_vote_bounds_not_imputation(self):
        import json
        from experiments.core.mechanism.upgrade_protocol_v3 import decode_content,majority
        def report(c):return decode_content(json.dumps({'A':'','E':[],'I':'','C':c}),['yes','no'])
        # Two matching votes determine majority irrespective of the missing seat.
        self.assertEqual({majority([report('yes'),report('yes'),report(c)],['yes','no'])
                          for c in ['yes','no','abstain']},{'yes'})
        # Split observed votes require retaining all possible conclusions.
        self.assertEqual({majority([report('yes'),report('no'),report(c)],['yes','no'])
                          for c in ['yes','no','abstain']},{'yes','no','abstain'})

    def test_literal_answer_compatibility(self):
        from experiments.mechanism.sensitivity_resource_allocation import explicit_answer
        self.assertEqual(explicit_answer('{"answer":"yes"}',['yes','no']),'yes')
        self.assertEqual(explicit_answer('{"C":"no","answer":"no"}',['yes','no']),'no')
        for value in ['{"C":"yes","answer":"no"}','{"C":"A"}',
                      '{"A":"the answer is yes"}','{"C":"yes","C":"no"}']:
            self.assertIsNone(explicit_answer(value,['yes','no']))


if __name__=='__main__':unittest.main()
