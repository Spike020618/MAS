import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.consensus_recommendation.common import create, fingerprint
from experiments.paired_deliberation.score_experiment import score


class PairedDevelopmentScoringTests(unittest.TestCase):
    def test_missing_cases_and_invalid_paths_keep_full_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create(root / "plan.json", {"case_ids": ["redial:1", "redial:2"]})
            good = {"status": "valid", "result": {
                "decision": {"ranking": ["m1"]},
                "ledger": {"prompt_tokens": 10, "completion_tokens": 2},
                "initial_reports": {"A": {"ranking": ["m1"]}},
                "events": [],
            }}
            record = {"case_id": "redial:1", "E1": {"single_serial": good},
                      "E2": {name: good for name in (
                          "peer_review", "self_review", "independent", "evidence_update")}}
            record["E2"]["peer_review"] = {"status": "invalid", "error": "fixture failure"}
            create(root / "cases/redial_1.json", {"record": record, "integrity": fingerprint(record)})

            def fixture_rows(path):
                if str(path).endswith("dev.labels.jsonl"):
                    return [{"case_id": "redial:1", "relevant_movie_ids": ["m1"]}]
                return [{"conversationId": 1, "initiatorWorkerId": "u1"}]

            with patch("experiments.paired_deliberation.score_experiment.read_rows", side_effect=fixture_rows):
                result = score(root)
            self.assertFalse(result["collection_complete"])
            self.assertEqual(result["missing_cases"], ["redial:2"])
            self.assertEqual(result["methods"]["E2:evidence_update"]["planned_denominator_hit_lower"], .5)
            self.assertEqual(result["methods"]["E2:peer_review"]["planned_denominator_hit_upper"], 1)
            self.assertFalse(result["formal_expansion_allowed"])
            self.assertFalse(result["reserved_test_read"])

    def test_tampered_case_is_rejected_before_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create(root / "plan.json", {"case_ids": ["redial:1"]})
            create(root / "cases/redial_1.json", {"record": {"case_id": "redial:1"}, "integrity": "wrong"})
            with patch("experiments.paired_deliberation.score_experiment.read_rows", return_value=[]):
                with self.assertRaisesRegex(ValueError, "integrity"):
                    score(root)


if __name__ == "__main__":
    unittest.main()
