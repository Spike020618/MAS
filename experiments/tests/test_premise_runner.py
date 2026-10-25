import json
from pathlib import Path
import tempfile
import unittest
from experiments.mechanism.run_premise_review import parse_row, vote, initial_request
from experiments.mechanism.run_upgrade_pilot import CallJournal


class PremiseRunnerTests(unittest.TestCase):
    def test_full_mock_execution_and_resume_without_new_calls(self):
        from unittest.mock import patch
        from contextlib import redirect_stdout
        import io
        from experiments.mechanism import run_premise_review as runner
        from experiments.core.llm import LLM_BACKENDS
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cases = [{'id': 'fixture:'+str(i), 'source_id': str(i), 'question': 'Q'+str(i),
                      'choices': ['yes', 'no']} for i in range(24)]
            (root/'public_cases.json').write_text(json.dumps(cases))
            (root/'private_labels.json').write_text(json.dumps([{'id': c['id'], 'label': 'yes'} for c in cases]))
            seen = []
            def transport(req):
                # No review call is allowed until ALL independent reports are sealed.
                is_review = 'sealed_reports' in req['messages'][1]['content']
                if is_review:
                    self.assertTrue((root/'sealed_initial.json').exists())
                    self.assertGreaterEqual(len(seen), 72)
                self.assertNotIn('gold', json.dumps(req))
                seen.append(req)
                return self.row()['response']
            cfg = {'api_key': 'fixture-not-a-secret', 'base_url': 'https://invalid.example', 'model': 'test-model'}
            with patch.object(runner, 'OUT', root), patch.object(runner, 'prepare'), \
                 patch.object(runner.p, 'transport', side_effect=transport), \
                 patch.dict(LLM_BACKENDS, {'deepseek': cfg}), redirect_stdout(io.StringIO()):
                runner.execute()
                self.assertEqual(len(seen), 120)
                runner.execute()
                self.assertEqual(len(seen), 120)
            result = json.loads((root/'result.json').read_text())
            self.assertEqual(result['initial_correct'], 24)
            self.assertEqual(result['actual_http_attempts'], 120)
            self.assertEqual(result['total_tokens_with_initial']['ordinary'], 24*4*30)

    def row(self, finish='stop', content=None):
        return {'response': {'http_status': 200, 'body': json.dumps({'model': 'test-model',
            'choices': [{'finish_reason': finish, 'message': {'content': content or '{"answer":"yes","reasoning":"no extra condition"}'}}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 20, 'total_tokens': 30}})}, 'wall_seconds': .5}

    def test_truncation_abstains_but_cost_remains(self):
        r = parse_row(self.row('length'), ['yes', 'no'])
        self.assertFalse(r['valid'])
        self.assertEqual(r['answer'], 'abstain')
        self.assertEqual(r['tokens'], 30)

    def test_unknown_usage_not_zero(self):
        r = self.row()
        b = json.loads(r['response']['body'])
        b.pop('usage')
        r['response']['body'] = json.dumps(b)
        self.assertIsNone(parse_row(r, ['yes', 'no'])['tokens'])

    def test_majority_requires_two(self):
        self.assertEqual(vote([{'answer': a} for a in ['yes', 'no', 'abstain']]), 'abstain')

    def test_initial_request_does_not_forward_label(self):
        c = {'question': 'Q', 'choices': ['yes', 'no'], 'gold': 'secret'}
        self.assertNotIn('secret', json.dumps(initial_request(c, 0, 'm')))

    def test_resume_no_second_transport_and_ambiguous_attempt_stops(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            seen = []
            journal = CallJournal(root, lambda r: (seen.append(r) or {'http_status': 200, 'body': '{}'}), 2)
            self.assertEqual(journal.call('k', {'a': 1}), journal.call('k', {'a': 1}))
            self.assertEqual(len(seen), 1)
            with self.assertRaises(ValueError):
                journal.call('k', {'a': 2})
        with tempfile.TemporaryDirectory() as d:
            journal = CallJournal(Path(d), lambda _: (_ for _ in ()).throw(TimeoutError()), 2)
            with self.assertRaises(RuntimeError):
                journal.call('k', {})
            with self.assertRaises(RuntimeError):
                journal.call('k', {})


if __name__ == '__main__':
    unittest.main()
