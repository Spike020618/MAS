import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments.mechanism import repeat_open_evidence_solver as r


class RepeatTests(unittest.TestCase):
    def test_primary_requests_have_no_answers(self):
        for c in r.primary.base.read(r.primary.OUT / 'public_cases.json'):
            for arm in r.ARMS:
                req, docs = r.original(c, arm)
                self.assertEqual(req, r.primary.request(c, arm, docs))

    def test_repeat_deduplicates_and_resumes_without_network(self):
        req = r.primary.request({'id': 'x', 'question': 'Example?', 'native_documents': []}, 'p01')
        body = {'model': 'deepseek-flash', 'system_fingerprint': r.primary.base.FP,
            'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
            'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(
                {'answer': 'abstain', 'reasoning': '', 'docids': []})}}]}
        count = []
        def transport(request):
            count.append(1); return {'http_status': 200, 'body': json.dumps(body)}
        with tempfile.TemporaryDirectory() as temp, patch.object(r, 'OUT', Path(temp)), patch.object(r, 'verify'):
            r.freeze(r.OUT / 'requests.json', {'x': {a: req for a in r.ARMS}})
            first = r.collect(transport)
            second = r.collect(lambda req: self.fail('resume API'))
            self.assertEqual(first, second); self.assertEqual(len(count), 1)

    def test_unresolved_attempt_blocks_alias(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(r, 'OUT', Path(temp)), patch.object(r, 'verify'):
            r.freeze(r.OUT / 'requests.json', {})
            (r.OUT / 'attempts.jsonl').write_text(json.dumps({'key': 'pending'}) + '\n')
            with self.assertRaisesRegex(RuntimeError, 'unresolved'):
                r.collect(lambda req: self.fail('unresolved API'))


if __name__ == '__main__': unittest.main()
