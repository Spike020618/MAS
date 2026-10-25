import json
from pathlib import Path
import tempfile
import unittest

from experiments.core.mechanism.upgrade_decode import is_failure, parse
from experiments.core.mechanism.upgrade_game import digest
from experiments.mechanism.calibrate_upgrade_v2 import audit_stage, collect, scored_rows


def response(report):
    return {'http_status': 200, 'requested_model': 'mock', 'body': json.dumps({
        'model': 'mock', 'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(report)}}],
        'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}})}


class DecodeTests(unittest.TestCase):
    def test_missing_inference_is_failure_not_fabricated_reasoning(self):
        row = {'response': response({'A': 'a', 'E': [], 'C': 'yes'})}
        report, usage = parse(row, ['yes', 'no'])
        self.assertTrue(is_failure(report))
        self.assertEqual(report['C'], 'abstain')
        self.assertEqual(usage['total_tokens'], 2)

    def test_valid_reports_unchanged(self):
        expected = {'A': '', 'E': [], 'I': '', 'C': 'yes'}
        self.assertEqual(parse({'response': response(expected)}, ['yes', 'no'])[0], expected)

    def test_transport_and_truncation_not_hidden(self):
        r = response({}); r['http_status'] = 500
        with self.assertRaises(ValueError): parse({'response': r}, ['yes', 'no'])
        r = response({}); b = json.loads(r['body']); b['choices'][0]['finish_reason'] = 'length'; r['body'] = json.dumps(b)
        with self.assertRaises(ValueError): parse({'response': r}, ['yes', 'no'])

    def test_failed_slots_keep_question_and_cost_without_retry(self):
        cases = [{'id': 'mock:1', 'source_id': '1', 'question': 'Fixture?', 'choices': ['yes', 'no']}]
        labels = [{'id': 'mock:1', 'label': 'yes'}]
        provenance = {'scope': 'test_only', 'public_hashes': {'calibration': digest(cases)},
                      'private_hashes': {'calibration': digest(labels)}, 'history_scan_hash': 'fixture'}
        calls = []

        def transport(request):
            calls.append(1)
            return response({'C': 'yes'})

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'collection'
            result = collect(cases, directory, provenance, 'calibration', None, transport, {'mock': True})
            self.assertEqual(result['invalid_report_slots'], 6)
            self.assertEqual(len(calls), 6)
            self.assertEqual(audit_stage(directory)['questions'], 1)
            path = Path(tmp) / 'labels.json'; path.write_text(json.dumps(labels))
            scored = scored_rows(directory, path)
            self.assertEqual(scored[0]['scores'], [0] * 8)
            self.assertEqual(scored[0]['initial_failures'], 3)


if __name__ == '__main__': unittest.main()
