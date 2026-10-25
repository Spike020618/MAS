from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
from experiments.mechanism import run_sampling_coordination_screen as r


class SamplingScreenTests(unittest.TestCase):
    def setUp(self):
        self.case = {'id': 'synthetic:sampling', 'question': 'Which?', 'choices': ['A', 'B'],
                     'source_id': 'synthetic', 'split': 'calibration'}
        self.calls = []

    def fake(self, req):
        self.calls.append(req)
        return {'http_status': 200, 'body': json.dumps({'model': 'deepseek-flash', 'system_fingerprint': r.base.FP,
            'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120},
            'choices': [{'finish_reason': 'stop', 'message': {'content': '{"answer":"A","reasoning":"test"}'}}]})}

    def slots(self, varying):
        return {f'{i}/{a}': {'answer': 'B' if varying and a == 2 else 'A', 'valid': True, 'tokens': 100}
                for i in range(3) for a in range(3)}

    def test_same_prompt_real_parameter_changes_no_labels(self):
        requests = [r.request(self.case, 0, a) for a in range(3)]
        self.assertEqual([v['temperature'] for v in requests], [.2, .8, 1.3])
        self.assertEqual(len({json.dumps(v['messages']) for v in requests}), 1)
        with self.assertRaises(ValueError): r.request({**self.case, 'label': 'A'}, 0, 0)

    def test_gate_rejects_no_action_effect(self):
        out = {'a': self.slots(False), 'b': self.slots(False)}
        gate = r.action_gate(out, {'a': 'A', 'b': 'A'})
        self.assertFalse(gate['passed'])

    def test_gate_requires_more_than_one_varying_question(self):
        self.assertFalse(r.action_gate({'a': self.slots(True)}, {'a': 'A'})['passed'])
        self.assertTrue(r.action_gate({'a': self.slots(True), 'b': self.slots(True)}, {'a': 'A', 'b': 'A'})['passed'])

    def test_full_ten_slots_and_zero_call_resume(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(r, 'OUT', Path(tmp)), \
             patch('socket.socket.connect', side_effect=AssertionError('offline')):
            out = r.collect(self.case, 'train', self.fake, {'used': 0, 'cap': 5})
            self.assertEqual(len(out), 10)
            self.assertEqual(out, r.collect(self.case, 'train', self.fake, {'used': 0, 'cap': 5}))
            self.assertEqual(len(self.calls), 10)

    def test_budget_and_unknown_timeout_stop(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(r, 'OUT', Path(tmp)):
            with self.assertRaisesRegex(RuntimeError, 'budget'):
                r.collect(self.case, 'train', self.fake, {'used': 5, 'cap': 5})
            self.assertFalse(self.calls)
            def bad(req): raise TimeoutError()
            with self.assertRaises(RuntimeError): r.collect(self.case, 'train', bad, {'used': 0, 'cap': 5})
            with self.assertRaisesRegex(RuntimeError, 'unresolved attempt'):
                r.collect(self.case, 'train', self.fake, {'used': 0, 'cap': 5})

    def test_selected_profile_is_committed_before_test_calls(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(r, 'OUT', Path(tmp)):
            policy = {'probabilities': [[0, 0, 1]] * 3}
            r.freeze(r.OUT / 'frozen_policy.json', policy)
            def checking(req):
                self.assertTrue(list(r.OUT.rglob('selected_before_calls.json')))
                return self.fake(req)
            r.collect(self.case, 'test', checking, {'used': 0, 'cap': 5}, policy)
            self.assertEqual([x['temperature'] for x in self.calls[:3]], [1.3] * 3)

    def test_cached_training_is_frozen_and_not_equilibrium_claim(self):
        out = {'a': self.slots(True), 'b': self.slots(True)}
        policy = r.train_policy(out, {'a': 'A', 'b': 'A'})
        self.assertTrue(policy['frozen'])
        self.assertEqual(policy['steps'], 3000)
        self.assertEqual(len(policy['train_empirical_deviation']), 3)


if __name__ == '__main__': unittest.main()
