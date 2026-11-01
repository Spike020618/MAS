import unittest
from experiments.core.mechanism import action_target_interface_v2 as v2


class ContractRepairTests(unittest.TestCase):
    docs = [{'docid': 'd', 'title': 'Title', 'text': 'Alice designed the ride.'}]

    def test_none_may_cite_real_support(self):
        report = {'status': 'none', 'document_id': 'd', 'quote': 'Alice designed', 'note': 'No mismatch.'}
        self.assertEqual(v2.normalize_report('check', report, self.docs), report)

    def test_none_may_have_no_citation(self):
        report = {'status': 'none', 'document_id': None, 'quote': None, 'note': 'No mismatch.'}
        self.assertEqual(v2.normalize_report('check', report, self.docs), report)

    def test_mismatch_still_requires_source(self):
        report = {'status': 'mismatch', 'document_id': None, 'quote': None, 'note': 'Disagree.'}
        self.assertIsNone(v2.normalize_report('check', report, self.docs))

    def test_unknown_quote_rejected_for_none(self):
        report = {'status': 'none', 'document_id': 'd', 'quote': 'Bob designed', 'note': 'No mismatch.'}
        self.assertIsNone(v2.normalize_report('check', report, self.docs))

    def test_review_extra_self_score_dropped(self):
        report = {'note': 'The answer matches the passage.', 'correct': True}
        self.assertEqual(v2.normalize_report('review', report, self.docs), {'note': report['note']})

    def test_word_cap_not_removed(self):
        self.assertIsNone(v2.normalize_report('review', {'note': 'word ' * 121}, self.docs))

    def test_prompt_and_validator_agree(self):
        case = {'question': 'Who?'}
        initial = {'answer': 'Alice', 'reasoning': 'Source.', 'docids': ['d']}
        req = v2.action_request('check', case, self.docs, initial)
        self.assertIn('you may cite a real supplied passage', req['messages'][0]['content'])
        initial_req = v2.final_request(case, self.docs)
        final_req = v2.final_request(case, self.docs, initial, {'note': 'Review'})
        self.assertEqual(initial_req['messages'][0], final_req['messages'][0])


if __name__ == '__main__':
    unittest.main()
