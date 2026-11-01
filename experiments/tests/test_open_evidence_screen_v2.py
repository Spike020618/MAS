from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments.mechanism import run_open_evidence_screen_v2 as r


class FakeCorpus:
    def __init__(self, path): self.count = 0
    def search(self, query, k=2):
        self.count += 1
        text = 'Evidence for ' + query
        return [{'docid': sha256(query.encode()).hexdigest()[:10], 'title': query, 'text': text,
                 'text_sha256': sha256(text.encode()).hexdigest(), 'rank_score': -1.}]
    def close(self): pass


class OpenEvidenceTests(unittest.TestCase):
    def setUp(self):
        text = 'Native source.'
        self.case = {'id': 'synthetic:open', 'question': 'Which connection?', 'native_documents': [
            {'docid': 'native:1', 'title': 'Native', 'text': text, 'text_sha256': sha256(text.encode()).hexdigest()}]}
        self.calls = []

    def fake(self, request):
        self.calls.append(request)
        prompt = request['messages'][0]['content']
        if 'exactly TWO' in prompt: value = {'queries': ['first entity', 'second entity']}
        elif 'ONE next search' in prompt: value = {'query': 'bridge from source'}
        else: value = {'answer': 'test', 'reasoning': 'reason', 'docids': []}
        return {'http_status': 200, 'body': json.dumps({'model': 'deepseek-flash', 'system_fingerprint': r.base.FP,
            'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120},
            'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(value)}}]})}

    def test_full_dag_and_cache_resume_no_new_model_calls(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(r, 'OUT', Path(tmp)), patch.object(r, 'Corpus', FakeCorpus), \
             patch('socket.socket.connect', side_effect=AssertionError('offline')):
            a = r.collect(self.case, self.fake, {'used': 0, 'cap': 5})
            b = r.collect(self.case, self.fake, {'used': 0, 'cap': 5})
            self.assertEqual(a, b); self.assertEqual(len(self.calls), 13)
            retrieval = r.base.read(Path(tmp) / self.case['id'] / 'retrieval.json')
            self.assertEqual(len(retrieval['environments']['p01']['searches']), 2)
            self.assertEqual(len(retrieval['environments']['s0']['searches']), 2)
            self.assertTrue(retrieval['environments']['s0']['searches'][1]['duplicate_ids'])

    def test_identical_evidence_does_not_create_new_stochastic_payoff(self):
        class ConstantCorpus(FakeCorpus):
            def search(self, query, k=2): return super().search('same article', k)
        with tempfile.TemporaryDirectory() as tmp, patch.object(r, 'OUT', Path(tmp)), patch.object(r, 'Corpus', ConstantCorpus):
            result = r.collect(self.case, self.fake, {'used': 0, 'cap': 5})
            self.assertLess(len(self.calls), 13)
            self.assertEqual(result['s0'], result['p01'])
            self.assertTrue(list(Path(tmp).rglob('*_alias.json')))

    def test_request_permissions_and_fixed_final_prompt(self):
        with self.assertRaises(ValueError): r.request({**self.case, 'answer': 'gold'}, 'direct')
        with self.assertRaises(ValueError): r.request(self.case, 'plan', self.case['native_documents'])
        cases = [r.request(self.case, a, self.case['native_documents']) for a in ('s0', 'p01', 'sequential', 'native')]
        self.assertEqual(len({q['messages'][0]['content'] for q in cases}), 1)
        for q in cases: self.assertEqual(set(json.loads(q['messages'][1]['content'])), {'question', 'documents'})

    def test_source_fabrication_rejected(self):
        doc = {**self.case['native_documents'][0], 'text': 'changed'}
        with self.assertRaises(ValueError): r.request(self.case, 'p01', [doc])

    def test_budget_checked_before_transport(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(r, 'OUT', Path(tmp)):
            with self.assertRaises(RuntimeError): r.collect(self.case, self.fake, {'used': 5, 'cap': 5})
            self.assertFalse(self.calls)

    def test_unknown_timeout_not_retried(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(r, 'OUT', Path(tmp)):
            def bad(req): raise TimeoutError()
            with self.assertRaises(RuntimeError): r.collect(self.case, bad, {'used': 0, 'cap': 5})
            with self.assertRaisesRegex(RuntimeError, 'unresolved attempt'):
                r.collect(self.case, self.fake, {'used': 0, 'cap': 5})
            self.assertFalse(self.calls)

    def test_plan_failure_is_explicit_not_repaired(self):
        row = {'response': self.fake(r.request(self.case, 'direct')), 'wall_seconds': 1}
        parsed = r.parse(row, 'plan', [])
        self.assertFalse(parsed['valid']); self.assertNotIn('queries', parsed)

    def test_reference_ids_not_used_as_correctness_oracle(self):
        row = {'response': self.fake(r.request(self.case, 'direct')), 'wall_seconds': 1}
        body = json.loads(row['response']['body'])
        body['choices'][0]['message']['content'] = json.dumps({'answer': 'test', 'reasoning': 'r', 'docids': ['fake']})
        row['response']['body'] = json.dumps(body)
        parsed = r.parse(row, 's0', [])
        self.assertTrue(parsed['valid']); self.assertFalse(parsed['document_ids_registered'])


if __name__ == '__main__': unittest.main()

