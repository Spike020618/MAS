import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments.mechanism import run_action_comparison as r
from experiments.mechanism.prepare_action_comparison import select
from experiments.mechanism.run_upgrade_pilot import freeze
from experiments.mechanism.score_action_comparison import exact_p


class FakeCorpus:
    closed = 0
    def search(self, query, k):
        text = 'Alice designed the ride.'
        return [{'docid': 'original2', 'title': 'Second', 'text': text, 'text_sha256': r.sha_text(text)}]
    def close(self):
        FakeCorpus.closed += 1


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        text = 'Alice designed the ride.'
        self.case = {'id': 'abc123', 'question': 'Who designed the ride?', 'native_documents': [
            {'docid': 'original', 'title': 'Original', 'text': text, 'text_sha256': r.sha_text(text)}]}
        self.prices = {'input_usd_per_million': 0.3, 'output_usd_per_million': 1.2}
        self.calls = []

    def transport(self, req):
        self.calls.append(req)
        p = json.loads(req['messages'][1]['content'])
        system = req['messages'][0]['content']
        if 'Produce one search query' in system:
            value = {'query': 'Alice ride design'}
        elif 'Review the initial answer' in system:
            value = {'note': 'The source supports Alice.'}
        elif 'Check one most consequential' in system:
            value = {'status': 'mismatch', 'document_id': p['documents'][0]['docid'],
                     'quote': 'Alice designed', 'note': 'Use the explicit subject.'}
        else:
            value = {'reasoning': 'The document says Alice.', 'docids': [p['documents'][0]['docid']], 'answer': 'Alice'}
        return {'http_status': 200, 'body': json.dumps({'model': r.interface.MODEL, 'system_fingerprint': r.FP,
            'usage': {'prompt_tokens': 30, 'completion_tokens': 20, 'total_tokens': 50},
            'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(value)}}]})}

    def collect(self, **kwargs):
        return r.collect(self.case, self.out, kwargs.pop('transport', self.transport),
                         self.prices, FakeCorpus, **kwargs)

    def test_full_collect_and_zero_call_resume(self):
        sealed = self.collect()
        self.assertEqual(len(self.calls), 8)
        self.assertEqual(set(sealed['arms']), set(r.ARMS))
        def forbidden(req):
            raise AssertionError('resume called API')
        self.assertEqual(sealed, self.collect(transport=forbidden))
        r.replay_case(self.case, sealed, r.read_rows(self.out / self.case['id'] / 'responses.jsonl'))

    def test_replay_rejects_cross_branch_request(self):
        sealed = self.collect()
        rows = r.read_rows(self.out / self.case['id'] / 'responses.jsonl')
        row = next(v for v in rows if v['key'] == 'check/action')
        row['request']['messages'][1]['content'] = '{}'
        with self.assertRaisesRegex(ValueError, 'permissions'):
            r.replay_case(self.case, sealed, rows)

    def test_no_cross_arm_information(self):
        self.collect()
        for req in self.calls:
            p = json.loads(req['messages'][1]['content'])
            self.assertNotIn('gold', p)
            self.assertNotIn('supporting_facts', p)
            self.assertNotIn('arms', p)
            if 'note' not in p:  # action request
                self.assertEqual(len(p['documents']), 1)

    def test_unknown_transport_blocks_resume(self):
        def lost(req):
            raise TimeoutError('simulated unknown paid outcome')
        with self.assertRaises(RuntimeError):
            self.collect(transport=lost)
        with self.assertRaisesRegex(RuntimeError, 'unknown pending'):
            self.collect()
        self.assertEqual(len(self.calls), 0)

    def test_budget_checked_before_transport(self):
        with self.assertRaisesRegex(RuntimeError, 'budget'):
            self.collect(cap=0.000001)
        self.assertFalse(self.calls)

    def test_call_cap_before_transport(self):
        with self.assertRaisesRegex(RuntimeError, 'request cap'):
            self.collect(maximum_calls=0)
        self.assertFalse(self.calls)

    def test_invalid_notes_dropped_without_retry(self):
        def invalid_note(req):
            raw = self.transport(req)
            system = req['messages'][0]['content']
            if 'Check one most consequential' in system:
                body = json.loads(raw['body'])
                body['choices'][0]['message']['content'] = '{}'
                raw['body'] = json.dumps(body)
            return raw
        sealed = self.collect(transport=invalid_note)
        self.assertTrue(sealed['arms']['check']['invalid_action_note_dropped'])
        self.assertEqual(sealed['arms']['check']['final']['source_key'], sealed['arms']['resample']['final']['source_key'])
        self.assertEqual(len(self.calls), 7)

    def test_backend_change_stops(self):
        def changed(req):
            raw = self.transport(req)
            body = json.loads(raw['body']); body['system_fingerprint'] = 'changed'
            raw['body'] = json.dumps(body)
            return raw
        with self.assertRaisesRegex(RuntimeError, 'identity'):
            self.collect(transport=changed)
        self.assertEqual(len(self.calls), 1)
        with self.assertRaises(RuntimeError):
            self.collect()

    def test_tampered_journal_stops(self):
        self.collect()
        path = self.out / self.case['id'] / 'responses.jsonl'
        rows = r.read_rows(path); rows[0]['request']['model'] = 'tampered'
        path.write_text(''.join(json.dumps(v) + '\n' for v in rows))
        with self.assertRaisesRegex(ValueError, 'altered'):
            self.collect()

    def test_private_fields_rejected(self):
        self.case['gold'] = 'secret'
        with self.assertRaises(ValueError):
            self.collect()
        self.assertFalse(self.calls)

    def test_prices_fail_closed(self):
        prices = {**self.prices, 'checked_utc_date': '2026-10-25', 'official_url': 'https://api-docs.deepseek.com/quick_start/pricing'}
        r.valid_prices(prices, today='2026-10-25')
        with self.assertRaises(ValueError):
            r.valid_prices(prices, today='2026-10-26')
        prices['input_usd_per_million'] = float('nan')
        with self.assertRaises(ValueError):
            r.valid_prices(prices, today='2026-10-25')

    def test_truncation_recorded_not_retried(self):
        def truncated(req):
            raw = self.transport(req)
            body = json.loads(raw['body'])
            body['choices'][0]['finish_reason'] = 'length'
            raw['body'] = json.dumps(body)
            return raw
        sealed = self.collect(transport=truncated)
        self.assertFalse(sealed['initial']['parsed']['valid'])
        self.assertFalse(sealed['arms']['check']['final']['parsed']['valid'])
        self.assertLessEqual(len(self.calls), 8)

    def test_missing_attempt_rejected(self):
        self.collect()
        (self.out / self.case['id'] / 'attempts.jsonl').unlink()
        with self.assertRaisesRegex(RuntimeError, 'unknown pending'):
            self.collect()

    def test_scoring_refuses_incomplete_seal_before_labels(self):
        from experiments.mechanism import score_action_comparison as scorer
        reads = []
        def read(path):
            reads.append(path.name)
            if path.name == 'public_cases.json':
                return [self.case]
            if path.name == 'sealed_outputs.json':
                return {}
            raise AssertionError('unexpected read before completeness check')
        with patch.object(r, 'verify'), patch.object(r, 'read', side_effect=read):
            with self.assertRaisesRegex(ValueError, 'all24'):
                scorer.main()
        self.assertNotIn('private_labels.json', reads)

    def test_complete_synthetic_score_is_not_efficacy(self):
        from experiments.mechanism import score_action_comparison as scorer
        cases, outputs, labels, splits = [], {}, {}, {}
        for i in range(24):
            case = {**self.case, 'id': 'fake' + str(i)}
            cases.append(case)
            outputs[case['id']] = r.collect(case, self.out, self.transport, self.prices, FakeCorpus)
            labels[case['id']] = {'answer': 'Alice'}
            splits[case['id']] = 'development' if i < 12 else 'confirmation'
        for name, value in [('public_cases.json', cases), ('sealed_outputs.json', outputs),
                            ('private_labels.json', labels), ('splits.json', splits)]:
            freeze(self.out / name, value)
        freeze(self.out / 'selection.json', {'labels_sha256': r.sha(self.out / 'private_labels.json')})
        freeze(self.out / 'collection_receipt.json', {'calls': 192, 'tokens': 192 * 50, 'usd_proxy': 0,
            'sealed_outputs_sha256': r.sha(self.out / 'sealed_outputs.json')})
        with patch.object(r, 'OUT', self.out), patch.object(r, 'verify'), patch('builtins.print'):
            scorer.main()
        result = r.read(self.out / 'analysis.json')
        self.assertEqual(result['scientific_gate'], 'FAILED_no_expansion')
        self.assertEqual(result['cells']['confirmation']['arms']['check']['correct'], 12)


class SelectionTests(unittest.TestCase):
    @staticmethod
    def row(cid, titles):
        return {'id': cid, 'question': cid, 'supporting_facts': {'title': titles}}

    def test_transitive_historical_page_exclusion(self):
        rows = [self.row('old', ['A']), self.row('bridge', ['A', 'B']), self.row('candidate', ['B']),
                self.row('fresh1', ['C']), self.row('fresh2', ['D'])]
        selected, groups = select(rows, {'old'}, n=2)
        self.assertEqual({v['id'] for v in selected}, {'fresh1', 'fresh2'})
        self.assertEqual(len(set(groups.values())), 2)

    def test_insufficient_groups_not_relaxed(self):
        with self.assertRaisesRegex(ValueError, 'insufficient'):
            select([self.row('x', ['A']), self.row('y', ['A'])], set(), n=2)

    def test_exact_p(self):
        self.assertEqual(exact_p(0, 0), 1.0)
        self.assertAlmostEqual(exact_p(5, 0), 0.0625)


if __name__ == '__main__':
    unittest.main()
