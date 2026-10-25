import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from hashlib import sha256

from experiments.mechanism import run_full_context_challenge as run


class FullContextTests(unittest.TestCase):
    def setUp(self):
        article = 'The source establishes a fact. ' * 30
        self.case = {'id': 'synthetic:one', 'article_id': 'synthetic', 'article': article,
                     'article_sha256': sha256(article.encode()).hexdigest(), 'question': 'Which?',
                     'choices': dict(zip('ABCD', ('a', 'b', 'c', 'd')))}
        self.calls = []

    def fake(self, request):
        self.calls.append(request)
        body = {'answer': 'A', 'reasoning': 'source', 'argument': 'a supported argument',
                'challenge': 'Which premise supports it?', 'query': 'source fact',
                'verdict': 'reject', 'reason': 'unsupported', 'citations': []}
        return {'http_status': 200, 'body': json.dumps({'model': 'deepseek-flash',
            'system_fingerprint': run.FP, 'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120},
            'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(body)}}]})}

    def test_full_dag_resume_scoring_and_identical_final_interfaces(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch('socket.socket.connect', side_effect=AssertionError('offline')), \
             patch('socket.getaddrinfo', side_effect=AssertionError('offline')):
            out = Path(tmp)
            result = run.collect(self.case, out, self.fake, {'used': 0, 'cap': 20})
            self.assertEqual(len(self.calls), 45)
            for request in self.calls:
                data = json.loads(request['messages'][1]['content'])
                self.assertEqual(data['article'], self.case['article'])
                self.assertNotIn('gold_label', data)
            second = run.collect(self.case, out, self.fake, {'used': 0, 'cap': 20})
            self.assertEqual(result, second)
            self.assertEqual(len(self.calls), 45)
            # Both final instructions are identical, no arm name leaked.
            finals = [r for r in self.calls if 'response' in json.loads(r['messages'][1]['content']).get('history', {})]
            self.assertEqual(len(finals), 8)
            self.assertEqual(len({r['messages'][0]['content'] for r in finals}), 1)
            run.freeze(out / 'sealed_outputs.json', {self.case['id']: result})
            run.freeze(out / 'sealed_readouts.json', {self.case['id']: {k: {m: s[m]['verdict'] for m in
                ('ordinary0', 'ordinary2', 'fixed_judge', 'challenge_judge', 'nonadaptive', 'interactive')}
                for k, s in result['claims'].items()}})
            run.freeze(out / 'private_labels.json', {self.case['id']: 'A'})
            run.analyze(out)
            metrics = run.read(out / 'result.json')['metrics']
            self.assertEqual(metrics['interactive']['calls'], 16)
            self.assertEqual(metrics['interactive']['qa_abstain'], 1)
            self.assertEqual(metrics['native']['qa_correct'], 1)

    def test_no_approval_blocks_before_transport(self):
        with patch('socket.socket.connect', side_effect=AssertionError('offline')):
            with self.assertRaisesRegex(ValueError, 'No paid authorization'):
                run.run(Path('/nonexistent'), None)

    def test_budget_stops_before_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, 'budget'):
                run.collect(self.case, Path(tmp), self.fake, {'used': 0, 'cap': 0})
            self.assertEqual(len(self.calls), 0)

    def test_unknown_transport_stops_resume(self):
        def broken(request):
            raise TimeoutError()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                run.collect(self.case, Path(tmp), broken, {'used': 0, 'cap': 20})
            with self.assertRaisesRegex(RuntimeError, 'unresolved attempt'):
                run.collect(self.case, Path(tmp), self.fake, {'used': 0, 'cap': 20})
            self.assertEqual(len(self.calls), 0)

    def test_model_change_stops_with_raw_response(self):
        def changed(request):
            response = self.fake(request)
            body = json.loads(response['body']); body['system_fingerprint'] = 'changed'
            response['body'] = json.dumps(body)
            return response
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, 'backend changed'):
                run.collect(self.case, Path(tmp), changed, {'used': 0, 'cap': 20})
            self.assertEqual(len(self.calls), 1)
            self.assertEqual(len(list(Path(tmp).rglob('responses.jsonl'))), 1)

    def test_invalid_model_content_preserved_without_repair(self):
        row = {'wall_seconds': 1, 'response': self.fake({})}
        body = json.loads(row['response']['body'])
        body['choices'][0]['message']['content'] = 'broken JSON'
        row['response']['body'] = json.dumps(body)
        self.assertFalse(run.parse(row, 'interactive', [])['valid'])
        self.assertFalse(run.parse(row, 'native', [])['valid'])

    def test_label_and_cross_arm_and_source_injection_rejected(self):
        with self.assertRaises(ValueError):
            run.req({**self.case, 'gold': 'A'}, 'A', 'author', {}, [])
        with self.assertRaises(ValueError):
            run.req(self.case, 'A', 'restatement', {'author': 'x', 'challenge': 'y'}, [])
        with self.assertRaises(ValueError):
            run.req(self.case, 'A', 'fixed_judge', {'author': 'x'},
                    [{'start': 0, 'end': 3, 'text': 'bad'}])

    def test_no_silent_truncation_and_finite_history(self):
        with self.assertRaisesRegex(ValueError, 'oversized'):
            run.req(self.case, 'A', 'reply', {'author': 'a' * 12001, 'challenge': 'x'}, [])


if __name__ == '__main__':
    unittest.main()
