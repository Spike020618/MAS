import unittest

from experiments.paired_deliberation.conflict_verifier import DualEvidenceVerifier
from experiments.paired_deliberation.verified_state import VerificationError


class ConflictGateTests(unittest.TestCase):
    sources = {"message:0": {"text": "I want Mystery tonight."},
               "message:1": {"text": "I do not want Mystery tonight."}}

    def check(self, support, refute, negative=False):
        def judge(payload):
            return {"support_spans": support, "refute_spans": refute,
                    "reason": "Evidence-relative sides.", "usage": {}}
        claim = {"text": "The seeker " + ("does not want" if negative else "wants") + " Mystery tonight.",
                 "source_ids": list(self.sources)}
        return DualEvidenceVerifier(self.sources, judge).verify(claim)

    def test_both_sides_abstain_symmetrically(self):
        a = {"source_id": "message:0", "quote": self.sources["message:0"]["text"]}
        b = {"source_id": "message:1", "quote": self.sources["message:1"]["text"]}
        for negative, support, refute in [(False, [a], [b]), (True, [b], [a])]:
            r = self.check(support, refute, negative)
            self.assertEqual(r["verdict"], "insufficient")
            self.assertTrue(r["evidence_conflict"])

    def test_absence_abstains_and_one_side_derives_verdict(self):
        a = {"source_id": "message:0", "quote": self.sources["message:0"]["text"]}
        self.assertEqual(self.check([], [])["verdict"], "insufficient")
        self.assertEqual(self.check([a], [])["verdict"], "supported")
        self.assertEqual(self.check([], [a])["verdict"], "refuted")

    def test_even_conflicting_output_cannot_hide_fabricated_quote(self):
        real = {"source_id": "message:0", "quote": self.sources["message:0"]["text"]}
        with self.assertRaises(VerificationError):
            self.check([real], [{"source_id": "message:1", "quote": "invented evidence"}])

    def test_gate_cannot_certify_model_evidence_completeness(self):
        # This is a documented limitation: omitting the opposite evidence can
        # still yield false support. Live symmetric probes remain mandatory.
        r = self.check([{"source_id": "message:0", "quote": self.sources["message:0"]["text"]}], [])
        self.assertEqual(r["verdict"], "supported")
        self.assertFalse(r["evidence_conflict"])


if __name__ == "__main__":
    unittest.main()
