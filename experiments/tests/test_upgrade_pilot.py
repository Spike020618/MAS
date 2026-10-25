import json
from pathlib import Path
import tempfile
import unittest

from experiments.mechanism.run_upgrade_pilot import (
    CallJournal, append, initial_request, parse, public_cases, read_rows, run,
)
from experiments.mechanism.audit_upgrade_pilot import audit
from experiments.mechanism.summarize_upgrade_pilot import summarize


CASES = [{'id': 'fixture-0', 'source_id': 'synthetic-unit-test-only',
          'question': 'Unit test fixture, not research data?', 'choices': ['yes', 'no']}]


def response(finish='stop', content=None):
    body = {'model': 'mock-no-api', 'choices': [{'finish_reason': finish, 'message': {
        'content': json.dumps({'A': '', 'E': [], 'I': '', 'C': 'yes'}) if content is None else content}}],
        'usage': {'prompt_tokens': 10, 'completion_tokens': 10, 'total_tokens': 20}}
    return {'http_status': 200, 'body': json.dumps(body), 'requested_model': 'mock'}


class PilotTests(unittest.TestCase):
    def test_full_resume_makes_no_new_calls_and_seals_before_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            calls = []

            def transport(request):
                # The fixture policy chooses one upgrade; last two are diagnostics.
                if len(calls) >= 4:
                    self.assertEqual(len(list(directory.glob('*.sealed.json'))), 1)
                if len(calls) >= 3:
                    self.assertEqual(len(list(directory.glob('*.selection.json'))), 1)
                calls.append(request)
                return response()

            first = run(CASES, directory, transport, {'mock': True})
            self.assertEqual(len(calls), 6)
            self.assertEqual(first['total_tokens'], 120)
            second = run(CASES, directory, transport, {'mock': True})
            self.assertEqual(first, second)
            self.assertEqual(len(calls), 6)
            self.assertEqual(len(read_rows(directory / 'attempts.jsonl')), 6)
            self.assertEqual(audit(directory)['status'], 'passed')
            diagnostics = summarize(directory)
            self.assertFalse(diagnostics['labels_opened'])
            self.assertEqual(diagnostics['responses'], 6)
            self.assertEqual(diagnostics['usage']['total_tokens'], 120)

    def test_unresolved_attempt_no_automatic_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            count = []

            def failing(request):
                count.append(1)
                raise TimeoutError('unknown paid outcome')

            journal = CallJournal(directory, failing)
            with self.assertRaises(RuntimeError): journal.call('job', {'x': 1})
            with self.assertRaises(RuntimeError): journal.call('job', {'x': 1})
            self.assertEqual(count, [1])

    def test_invalid_successful_response_is_retained_without_retry(self):
        for raw in (response(finish='length'), response(content='broken')):
            with tempfile.TemporaryDirectory() as tmp:
                count = []

                def transport(request):
                    count.append(1)
                    return raw

                directory = Path(tmp)
                for _ in range(2):
                    with self.assertRaises(ValueError): run(CASES, directory, transport, {'mock': True})
                self.assertEqual(count, [1])
                self.assertEqual(len(read_rows(directory / 'responses.jsonl')), 1)

    def test_after_one_persisted_response_reuses_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            request = initial_request(CASES[0], 0)
            CallJournal(directory, lambda _: response()).call('cached', request)
            restored = CallJournal(directory, lambda _: self.fail('must not call transport'))
            self.assertEqual(parse(restored.call('cached', request), ['yes', 'no'])[0]['C'], 'yes')
            with self.assertRaises(ValueError): restored.call('cached', {'changed': True})

    def test_budget_and_journal_corruption_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            journal = CallJournal(directory, lambda _: response(), 1)
            journal.call('1', {'a': 1})
            with self.assertRaises(RuntimeError): journal.call('2', {'a': 2})
            append(directory / 'responses.jsonl', read_rows(directory / 'responses.jsonl')[0])
            with self.assertRaises(ValueError): journal.call('1', {'a': 1})

    def test_label_in_public_case_rejected(self):
        for field in ('label', 'answer', 'correct_answer'):
            with self.assertRaises(ValueError): public_cases([{**CASES[0], field: 'yes'}])

    def test_protocol_drift_fails_before_transport(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            run(CASES, directory, lambda _: response(), {'mock': True})
            with self.assertRaises(ValueError):
                run(CASES, directory, lambda _: self.fail('must not execute'), {'mock': False})

    def test_raw_response_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            journal = CallJournal(directory, lambda _: response())
            journal.call('job', {'x': 1})
            path = directory / 'responses.jsonl'
            row = read_rows(path)[0]
            row['response']['http_status'] = 500
            # Test-only fixture mutation, not repair of a research journal.
            path.write_text(json.dumps(row) + '\n')
            with self.assertRaises(ValueError): journal.call('job', {'x': 1})

    def test_independent_audit_rejects_reordered_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            run(CASES, directory, lambda _: response(), {'mock': True})
            path = directory / 'timeline.jsonl'
            rows = read_rows(path)
            seal = next(i for i, r in enumerate(rows) if r['event'] == 'seal')
            diagnostic = next(i for i, r in enumerate(rows) if '/diagnostic/' in r['key'] and r['event'] == 'attempt')
            rows[seal], rows[diagnostic] = rows[diagnostic], rows[seal]
            for i, row in enumerate(rows): row['sequence'] = i
            path.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
            with self.assertRaises(ValueError): audit(directory)

    def test_independent_audit_rejects_unsupported_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            run(CASES, directory, lambda _: response(), {'mock': True})
            path = next(directory.glob('*.events.json'))
            value = json.loads(path.read_text())
            value[1]['report']['C'] = 'no'
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError): audit(directory)


if __name__ == '__main__': unittest.main()
