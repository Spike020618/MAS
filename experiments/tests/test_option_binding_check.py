from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments.mechanism import run_option_binding_check as run


class BindingTests(unittest.TestCase):
    def setUp(self):
        article = 'Alice found a book. Bob found a stone.'
        self.case = {'id': 'synthetic:binding', 'article_id': 'fake', 'article': article,
            'article_sha256': sha256(article.encode()).hexdigest(), 'question': 'What did Alice find?',
            'choices': {'A': 'book', 'B': 'stone', 'C': 'tree', 'D': 'boat'}}
        self.requests = []

    def fake(self, request):
        self.requests.append(request)
        instruction = request['messages'][0]['content']; data = json.loads(request['messages'][1]['content'])
        if 'Copy the original question' in instruction:
            v = {'question': data['question'], 'options': data['choices']}
        elif 'Check whether the initial reasoning' in instruction:
            v = {'findings': [], 'no_mismatch': True}
        elif 'Independently review' in instruction:
            v = {'review': 'No mismatch found.'}
        else:
            v = {'answer': 'A', 'reasoning': 'Alice found a book.'}
        return {'http_status': 200, 'body': json.dumps({'model': 'deepseek-flash', 'system_fingerprint': run.base.FP,
            'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120},
            'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(v)}}]})}

    def test_full_dag_resume_and_no_labels_or_arm_in_final(self):
        with tempfile.TemporaryDirectory() as tmp, patch('socket.socket.connect', side_effect=AssertionError('offline')):
            result = run.collect(self.case, Path(tmp), self.fake, {'used': 0, 'cap': 5})
            again = run.collect(self.case, Path(tmp), self.fake, {'used': 0, 'cap': 5})
            self.assertEqual(result, again); self.assertEqual(len(self.requests), 7)
            finals = [r for r in self.requests if 'review_report' in json.loads(r['messages'][1]['content'])]
            self.assertEqual(len(finals), 3)
            self.assertEqual(len({r['messages'][0]['content'] for r in finals}), 1)
            for r in self.requests:
                data = json.loads(r['messages'][1]['content'])
                self.assertEqual(data['article'], self.case['article'])
                self.assertNotIn('gold_label', data); self.assertNotIn('arm', data)

    def test_permissions(self):
        with self.assertRaises(ValueError): run.request({**self.case, 'label': 'A'}, 'initial')
        with self.assertRaises(ValueError): run.request(self.case, 'initial', {'answer': 'A', 'reasoning': 'x'})
        with self.assertRaises(ValueError): run.request(self.case, 'ordinary', {'answer': 'A', 'reasoning': 'x'}, 'cross-arm')
        with self.assertRaises(ValueError): run.request(self.case, 'final', {'answer': 'A', 'reasoning': 'x', 'truth': 'A'}, 'x')

    def test_budget_stops_before_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError): run.collect(self.case, Path(tmp), self.fake, {'used': 5, 'cap': 5})
            self.assertFalse(self.requests)

    def test_timeout_never_retried(self):
        def bad(req): raise TimeoutError()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError): run.collect(self.case, Path(tmp), bad, {'used': 0, 'cap': 5})
            with self.assertRaisesRegex(RuntimeError, 'unresolved attempt'):
                run.collect(self.case, Path(tmp), self.fake, {'used': 0, 'cap': 5})
            self.assertFalse(self.requests)

    def test_changed_backend_stops(self):
        def changed(req):
            r = self.fake(req); b = json.loads(r['body']); b['system_fingerprint'] = 'new'; r['body'] = json.dumps(b); return r
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, 'backend changed'):
                run.collect(self.case, Path(tmp), changed, {'used': 0, 'cap': 5})
            self.assertEqual(len(self.requests), 1)

    def row(self, content):
        r = self.fake(run.request(self.case, 'initial')); b = json.loads(r['body'])
        b['choices'][0]['message']['content'] = json.dumps(content); r['body'] = json.dumps(b)
        return {'response': r, 'wall_seconds': 1}

    def test_relist_must_be_verbatim(self):
        value = {'question': self.case['question'], 'options': {**self.case['choices'], 'A': 'a book'}}
        self.assertFalse(run.parse(self.row(value), 'relist', self.case)['valid'])

    def test_no_forced_mismatch_and_no_literal_match_oracle(self):
        value = {'findings': [], 'no_mismatch': True}
        self.assertTrue(run.parse(self.row(value), 'binding', self.case)['valid'])
        value = {'findings': [{'kind': 'subject', 'claim': 'Bob', 'quote': 'nonliteral quotation', 'explanation': 'different subject'}], 'no_mismatch': False}
        parsed = run.parse(self.row(value), 'binding', self.case)
        self.assertTrue(parsed['valid']); self.assertEqual(parsed['quote_matches'], [False])
        self.assertNotIn('quote_matches', parsed['substantive'])

    def test_reasoning_does_not_override_answer(self):
        parsed = run.parse(self.row({'answer': 'B', 'reasoning': 'A is correct'}), 'final', self.case)
        self.assertEqual(parsed['answer'], 'B')
        parsed = run.parse(self.row({'answer': 'other', 'reasoning': 'A is correct'}), 'final', self.case)
        self.assertFalse(parsed['valid'])

    def test_exact_p_and_accounting(self):
        self.assertEqual(run.exact_p(0, 0), 1)
        self.assertEqual(run.exact_p(5, 0), .0625)
        self.assertEqual(run.exact_p(2, 2), 1)


if __name__ == '__main__': unittest.main()
