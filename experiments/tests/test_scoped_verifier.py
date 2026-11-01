import unittest

from experiments.paired_deliberation.scoped_verifier import ScopedEvidenceVerifier
from experiments.paired_deliberation.verified_state import VerificationError


class ScopeTests(unittest.TestCase):
    sources = {"a": {"text": "I enjoyed A."}, "b": {"text": "I did not enjoy A."}}

    def check(self, support, refute, support_spans=None, refute_spans=None):
        obj = {"supports_full_claim": support, "refutes_full_claim": refute,
               "support_spans": support_spans if support_spans is not None else [{"source_id": "a", "quote": "I enjoyed A."}],
               "refute_spans": refute_spans if refute_spans is not None else [{"source_id": "b", "quote": "I did not enjoy A."}],
               "reason": "Scoped proxy judgement.", "usage": {}}
        return ScopedEvidenceVerifier(self.sources, lambda _: obj).verify({"text": "A claim.", "source_ids": ["a", "b"]})

    def test_related_quotes_do_not_certify_partial_claim(self):
        self.assertEqual(self.check(False, False, refute_spans=[])["verdict"], "insufficient")

    def test_utterance_is_not_refuted_by_another_utterance(self):
        self.assertEqual(self.check(True, False)["verdict"], "supported")

    def test_universal_counterexample_refutes_despite_one_agreeing_source(self):
        self.assertEqual(self.check(False, True)["verdict"], "refuted")

    def test_same_scope_full_conflict_stays_unresolved(self):
        r = self.check(True, True)
        self.assertEqual(r["verdict"], "insufficient")
        self.assertTrue(r["evidence_conflict"])

    def test_booleans_and_supporting_spans_are_not_filled_or_coerced(self):
        with self.assertRaises(VerificationError):
            self.check("false", False)
        with self.assertRaises(VerificationError):
            self.check(True, False, support_spans=[])


if __name__ == "__main__":
    unittest.main()
