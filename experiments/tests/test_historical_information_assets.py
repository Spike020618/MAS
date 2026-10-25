import unittest
from experiments.mechanism.audit_historical_information_assets import ROOT, CANDIDATES, inspect_function


class HistoricalInformationAssetsTest(unittest.TestCase):
    def test_label_derived_hint_classification_has_direct_code_basis(self):
        r = inspect_function(ROOT/'experiments/run_memory_diverse.py', 'generate_evidence_hints')
        self.assertIn('question_item["ground_truth"]', r['text'])
        self.assertIn('correct_hint', r['text'])
        self.assertIn('misleading_hint', r['text'])

    def test_peer_visibility_does_not_become_a_new_source(self):
        r = inspect_function(ROOT/'experiments/mechanism/collect.py', 'peer_context')
        self.assertIn('previous', r['text'])
        self.assertIn('x.report.E', r['text'])
        self.assertNotIn('requests.get', r['text'])

    def test_selected_candidates_are_actual_source_functions(self):
        self.assertEqual(len(CANDIDATES), 6)
        for rel, name, _, _ in CANDIDATES:
            r = inspect_function(ROOT/rel, name)
            self.assertGreaterEqual(r['start_line'], 1)
            self.assertGreaterEqual(r['end_line'], r['start_line'])
            self.assertEqual(len(r['source_sha256']), 64)

    def test_public_source_extraction_uses_snapshot_not_an_answer_key(self):
        r = inspect_function(ROOT/'experiments/mechanism/pilot_grounded_evidence.py', 'ranked_sources')
        self.assertIn('snapshot_path', r['text'])
        self.assertNotIn('ground_truth', r['text'])


if __name__ == '__main__': unittest.main()
