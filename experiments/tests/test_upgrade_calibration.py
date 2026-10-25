import json
from pathlib import Path
import tempfile
import unittest

from experiments.core.mechanism.upgrade_game import digest
from experiments.core.mechanism.upgrade_value import contract_for, fit, outcomes, state_key
from experiments.mechanism.calibrate_upgrade import audit_stage, collect, compare, scored_rows
from experiments.mechanism.prepare_upgrade_p2 import native_mc


def report(c):
    return {'A': '', 'E': [], 'I': '', 'C': c}


def mock(request):
    return {'http_status': 200, 'requested_model': 'mock', 'body': json.dumps({
        'model': 'mock', 'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(report('yes'))}}],
        'usage': {'prompt_tokens': 10, 'completion_tokens': 10, 'total_tokens': 20}})}


class CalibrationTests(unittest.TestCase):
    def test_answer_identity_invariance(self):
        self.assertEqual(state_key([report(v) for v in ['yes', 'yes', 'no']]),
                         state_key([report(v) for v in ['B', 'B', 'A']]))
        self.assertNotEqual(state_key([report(v) for v in ['yes', 'yes', 'no']]),
                            state_key([report(v) for v in ['yes', 'no', 'yes']]))

    def test_no_increment_means_no_forced_upgrade(self):
        rows = [{'id': str(i), 'state': '000', 'scores': [i % 2] * 8} for i in range(20)]
        model = fit(rows)
        c = contract_for(model, [report('yes')] * 3)
        from experiments.core.mechanism.upgrade_game import improve
        self.assertEqual(improve(c)['actions'], [0, 0, 0])

    def test_outcomes_are_potential_profile_not_oracle_selection(self):
        s = outcomes([report('no')] * 3, [report('yes')] * 3, ['yes', 'no'], 'yes')
        self.assertEqual(s, [0, 0, 0, 1, 0, 1, 1, 1])

    def test_mc_order_is_label_independent(self):
        a = native_mc('x', '1', 'Question?', ['one', 'two'], 0)
        b = native_mc('x', '1', 'Question?', ['one', 'two'], 1)
        self.assertEqual(a['question'], b['question'])
        self.assertNotEqual(a['label'], b['label'])

    def test_full_stage_recovery_and_isolated_scoring(self):
        cases = [{'id': 'test:1', 'source_id': '1', 'question': 'Fixture?', 'choices': ['yes', 'no']}]
        provenance = {'scope': 'test_only', 'public_hashes': {'calibration': digest(cases)},
                      'private_hashes': {'calibration': digest([{'id': 'test:1', 'label': 'yes'}])}, 'history_scan_hash': 'fixture'}
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'collection'
            collect(cases, directory, provenance, 'calibration', None, mock, {'mock': True})
            self.assertEqual(audit_stage(directory)['status'], 'passed')
            collect(cases, directory, provenance, 'calibration', None,
                    lambda _: self.fail('resume made paid call'), {'mock': True})
            labels = Path(tmp) / 'labels.json'; labels.write_text(json.dumps([{'id': 'test:1', 'label': 'yes'}]))
            rows = scored_rows(directory, labels)
            self.assertEqual(rows[0]['scores'], [1] * 8)

    def test_scoring_checks_collection_before_opening_label_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError) as error:
                scored_rows(Path(tmp), Path(tmp) / 'DO_NOT_OPEN_PRIVATE_LABELS')
            self.assertNotIn('DO_NOT_OPEN_PRIVATE_LABELS', str(error.exception))

    def test_flat_payoffs_fail_effect_gate(self):
        calibration = [{'id': 'cal:' + str(i), 'state': '000', 'scores': [1] * 8} for i in range(10)]
        model = fit(calibration)
        development = [{'id': 'dev:' + str(i), 'state': '000', 'scores': [1] * 8,
                        'initial': [report('yes')] * 3, 'initial_tokens': 30,
                        'upgrade_tokens': [10] * 3, 'component_wall_seconds': [1] * 6} for i in range(5)]
        result = compare(calibration, development, model)
        self.assertEqual(result['effect_gate'], 'STOP_no_qualifying_development_signal')
        self.assertIsNone(result['selected_lambda'])
        self.assertTrue(all(c['mechanism']['upgrades'] == 0 for c in result['candidates']))


if __name__ == '__main__': unittest.main()
