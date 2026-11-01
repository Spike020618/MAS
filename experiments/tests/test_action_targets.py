import json
import unittest
from experiments.core.mechanism import action_target_interface as a
from experiments.mechanism.audit_action_targets import support_presence, review_category


class ActionTargetsTests(unittest.TestCase):
    def setUp(self):
        self.case = {'question': 'Which one?', 'answer': 'SECRET', 'gold': 'SECRET',
                     'supporting_facts': 'SECRET', 'audit_category': 'SECRET'}
        self.docs = [{'docid': 'd', 'title': 'T', 'text': 'Alice designed the ride.', 'gold_match': 'SECRET'}]
        self.initial = {'answer': 'Bob', 'reasoning': 'A guess', 'docids': ['d'], 'score': 'SECRET'}

    def test_action_inputs_exclude_offline_labels(self):
        for action in a.TASKS:
            req = a.action_request(action, self.case, self.docs, self.initial)
            self.assertNotIn('SECRET', json.dumps(req))
            p = json.loads(req['messages'][1]['content'])
            self.assertEqual(set(p), {'question', 'documents', 'initial_answer'})

    def test_same_final_system_and_mode(self):
        direct = a.final_request(self.case, self.docs)
        upgraded = a.final_request(self.case, self.docs, self.initial, {'note': 'untrusted'})
        self.assertEqual(direct['messages'][0], upgraded['messages'][0])
        self.assertEqual(direct['thinking'], {'type': 'enabled'})
        self.assertEqual(direct['max_tokens'], 4096)
        self.assertNotIn('temperature', direct)
        self.assertNotIn('SECRET', json.dumps(upgraded))

    def test_no_unknown_actions(self):
        with self.assertRaises(ValueError):
            a.action_request('gold_checker', self.case, self.docs, self.initial)

    def test_literal_citation_is_not_semantic_proof(self):
        finding = {'status': 'mismatch', 'document_id': 'd', 'quote': 'Alice designed', 'note': 'claim'}
        self.assertTrue(a.check_finding(finding, self.docs))
        finding['quote'] = 'Bob designed'
        self.assertFalse(a.check_finding(finding, self.docs))
        finding['quote'] = ''
        self.assertFalse(a.check_finding(finding, self.docs))

    def test_no_forced_objection(self):
        finding = {'status': 'none', 'document_id': None, 'quote': None, 'note': 'No mismatch found.'}
        self.assertTrue(a.check_finding(finding, self.docs))
        finding['document_id'] = 'd'
        self.assertFalse(a.check_finding(finding, self.docs))

    def test_duplicate_sources_rejected(self):
        with self.assertRaises(ValueError):
            a.documents(self.docs * 2)

    def test_missing_support_not_equal_missing_truth(self):
        src = {'context': {'title': ['T'], 'sentences': [['Alice designed the ride.']]},
               'supporting_facts': {'title': ['T', 'missing'], 'sent_id': [0, 0]}}
        facts = support_presence(src, self.docs)
        self.assertEqual(facts[0]['matching_document_ids'], ['d'])
        self.assertIsNone(facts[1]['text'])
        self.assertEqual(facts[1]['matching_document_ids'], [])

    def test_unreviewed_error_fails_closed(self):
        with self.assertRaises(ValueError):
            review_category('unknown', 'enabled', {'em': 0})
        self.assertEqual(review_category('unknown', 'enabled', {'em': 1}), 'none')


if __name__ == '__main__':
    unittest.main()
