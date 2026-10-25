import unittest
from experiments.mechanism.analyze_real_qa_payoffs import recover_c, paired_summary


class RealQAPayoffAnalysisTests(unittest.TestCase):
    def test_intact_conclusion_is_recovered_without_completing_json(self):
        self.assertEqual(recover_c('{"I":broken, "C":"Yes"}'), 'Yes')
        self.assertIsNone(recover_c('{"C":"Ye'))
        self.assertIsNone(recover_c('{"C":"Yes","C":"No"}'))

    def test_incomplete_and_duplicate_pairs_are_rejected(self):
        row={'dataset':'d','question_id':'q','repeat':0,'action':'direct','correct':True}
        with self.assertRaises(ValueError): paired_summary([row])
        with self.assertRaises(ValueError): paired_summary([row,row])

    def test_repeats_stay_within_question_cluster(self):
        rows=[]
        for repeat in [0,1]:
            for action in ['direct','self_check']:
                rows.append({'dataset':'d','question_id':'q','repeat':repeat,
                             'action':action,'correct':action=='self_check'})
        summary=paired_summary(rows)
        self.assertEqual(summary['question_clusters'],1)
        self.assertEqual(summary['paired_repeat_units'],2)
        self.assertEqual(summary['accuracy_difference'],1)


if __name__=='__main__':unittest.main()
