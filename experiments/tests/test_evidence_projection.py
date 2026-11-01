import copy
import unittest

from experiments.consensus_recommendation.common import fingerprint
from experiments.paired_deliberation.evidence_projection_audit import project
from experiments.tests.test_design_v6 import evidence, report


class ProjectionTests(unittest.TestCase):
    def test_recorded_refutation_is_consumed_without_changing_input_or_repairing_proposal(self):
        initial = {"current_request": report()}
        before = copy.deepcopy(initial)
        check = {"verdict": "refuted", "kind": "fixture"}
        state, missing = project(evidence(), initial, [{"target": "current_request:c1"}],
                                 {fingerprint(initial["current_request"]["claims"][0]): check})
        self.assertEqual(state.claims["current_request:c1"]["status"], "retracted")
        self.assertEqual(initial, before)
        self.assertEqual(missing, [])

    def test_invalid_proxy_keeps_failure_flag_and_is_not_certified(self):
        initial = {"current_request": report()}
        check = {"verdict": "insufficient", "kind": "invalid_proxy", "verification_failed": True}
        state, _ = project(evidence(), initial, [{"target": "current_request:c1"}],
                           {fingerprint(initial["current_request"]["claims"][0]): check})
        self.assertEqual(state.claims["current_request:c1"]["status"], "unresolved")
        self.assertTrue(state.claims["current_request:c1"]["verification"]["verification_failed"])


if __name__ == "__main__":
    unittest.main()
