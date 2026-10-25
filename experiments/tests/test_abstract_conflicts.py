from collections import Counter
import json
import unittest
from experiments.mechanism.pilot_abstract_conflicts import evaluate,generate,validate_citations


class AbstractConflictTests(unittest.TestCase):
    def test_three_valued_truth_not_majority(self):
        self.assertEqual(evaluate(['enabled',None],'all'),'abstain')
        self.assertEqual(evaluate(['disabled',None],'all'),'no')
        self.assertEqual(evaluate(['enabled',None],'any'),'yes')
        self.assertEqual(evaluate(['disabled',None],'any'),'abstain')
        self.assertEqual(evaluate(['enabled',None],'at_least'),'abstain')
        self.assertEqual(evaluate(['disabled',None],'at_least'),'no')

    def test_determinism_balance_and_split_separation(self):
        public,private=generate();self.assertEqual((public,private),generate())
        self.assertEqual(len({r['question_id'] for r in public}),12)
        for split in ['calibration','heldout_seed']:
            subset=[r for r in private if r['split']==split]
            self.assertEqual(Counter(r['ground_truth'] for r in subset),{'yes':2,'no':2,'abstain':2})
        self.assertEqual([r['question_id'] for r in public],[r['question_id'] for r in private])

    def test_public_packets_have_no_scoring_or_preassigned_answers(self):
        public,private=generate()
        for case in public:
            text=json.dumps(case)
            for forbidden in ['ground_truth','target_statuses','wanted','initial_C','preferred_answer']:
                self.assertNotIn(forbidden,text)
            self.assertEqual(len(case['local_documents']),3)
            self.assertEqual(len(case['paid_documents']),2)
        for row in private:
            self.assertEqual(evaluate(row['target_statuses'],row['operator']),row['ground_truth'])

    def test_private_scorer_order_invariant(self):
        _,private=generate()
        for row in private:
            self.assertEqual(evaluate(row['target_statuses'][::-1],row['operator']),row['ground_truth'])

    def test_quotes_are_source_relative_not_truth(self):
        source={'source_id':'s','text':'A sufficiently long literal statement.'}
        row={'parsed':{'citations':[{'source_id':'s','quote':'sufficiently long literal'}]}}
        self.assertEqual(validate_citations(row,[source])['valid'],1)
        row['parsed']['citations'][0]['source_id']='answer_key'
        self.assertEqual(validate_citations(row,[source])['invalid'],1)

    def test_truth_against_independent_completion_oracle(self):
        from itertools import product
        for statuses in product(['enabled','disabled',None],repeat=4):
            missing=[i for i,v in enumerate(statuses) if v is None]
            for operator in ['all','any','at_least']:
                possible=set()
                for fills in product([False,True],repeat=len(missing)):
                    assignment=[v=='enabled' for v in statuses]
                    for i,value in zip(missing,fills):assignment[i]=value
                    decision=all(assignment) if operator=='all' else any(assignment) if operator=='any' else sum(assignment)>=2
                    possible.add(decision)
                expected='yes' if possible=={True} else 'no' if possible=={False} else 'abstain'
                self.assertEqual(evaluate(statuses,operator),expected)

    def test_bad_dispute_shape_is_recordable_not_a_crash(self):
        from experiments.mechanism.pilot_abstract_conflicts import valid_dispute
        self.assertFalse(valid_dispute({'parsed':[]}))
        self.assertFalse(valid_dispute({'parsed':None}))
        self.assertFalse(valid_dispute({'parsed':{'C':'yes'}}))

    def test_parse_failure_abstention_not_valid_evidence_of_unknown(self):
        from experiments.mechanism.audit_abstract_conflicts import valid_majority_support
        good={'parsed':{'C':'abstain','A':'a','E':'e','I':'i'}}
        bad={'parsed':None}
        self.assertFalse(valid_majority_support([bad,bad,bad],'abstain'))
        self.assertFalse(valid_majority_support([good,bad,bad],'abstain'))
        self.assertTrue(valid_majority_support([good,good,bad],'abstain'))


if __name__=='__main__':unittest.main()
