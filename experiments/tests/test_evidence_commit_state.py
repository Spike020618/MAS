import copy
import unittest

from experiments.consensus_recommendation.common import fingerprint
from experiments.paired_deliberation.evidence_commit_state import EvidenceCommitState
from experiments.paired_deliberation.state_v6 import ReplacementPreconditionError
from experiments.paired_deliberation.verified_state import EvidenceVerifier
from experiments.tests.test_design_v6 import evidence, report, claim


class CommitTests(unittest.TestCase):
    def setUp(self):
        self.e = evidence()
        self.verifier = EvidenceVerifier({"movie:1": self.e["candidates_in_shared_retrieval_order"][0]})
        self.target = "current_request:c1"

    def state(self, correct=False):
        return EvidenceCommitState({"current_request": report([
            claim(text="Movie 1 has catalog year %d." % (2000 if correct else 2001))])}, self.e)

    def observe(self, s):
        return s.observe("current_request", self.target, fingerprint(s.claims[self.target]), self.verifier)

    def proposal(self, s, replacement=None):
        return {"target_claim_id": self.target, "expected_hash": fingerprint(s.claims[self.target]), "replacement": replacement}

    def test_bad_repair_does_not_erase_refutation(self):
        s = self.state()
        self.observe(s)
        before = copy.deepcopy(s.claims)
        invalid = claim(text="Movie 1 has catalog year 2000.", stance="refute")
        with self.assertRaises(ValueError):
            s.propose("current_request", self.target, self.proposal(s, invalid), self.verifier)
        self.assertEqual(s.claims, before)
        self.assertEqual(s.claims[self.target]["status"], "retracted")

    def test_unnecessary_repair_does_not_erase_support(self):
        s = self.state(correct=True)
        self.observe(s)
        with self.assertRaises(ReplacementPreconditionError):
            s.propose("current_request", self.target, self.proposal(s, claim(text="Movie 1 has catalog year 2000.")), self.verifier)
        self.assertEqual(s.claims[self.target]["status"], "supported")

    def test_supported_replacement_is_a_second_transaction(self):
        s = self.state()
        self.observe(s)
        s.propose("current_request", self.target, self.proposal(s, claim(text="Movie 1 has catalog year 2000.")), self.verifier)
        self.assertEqual(s.claims[self.target]["status"], "supported")
        self.assertEqual(s.claims[self.target]["revision"], 2)
        self.assertEqual([e["action"] for e in s.events], ["observe_retract", "replace"])

    def test_stale_pre_observation_hash_is_not_silently_rebound(self):
        s = self.state()
        p = self.proposal(s)
        self.observe(s)
        with self.assertRaisesRegex(ValueError, "stale"):
            s.propose("current_request", self.target, p, self.verifier)
        self.assertEqual(s.claims[self.target]["status"], "retracted")

    def test_unsupported_replacement_preserves_old_retraction(self):
        s = self.state()
        self.observe(s)
        with self.assertRaises(ReplacementPreconditionError):
            s.propose("current_request", self.target, self.proposal(s, claim(text="Movie 1 has catalog year 1999.")), self.verifier)
        self.assertEqual(s.claims[self.target]["status"], "retracted")

    def test_observation_is_idempotent_and_legacy_path_is_disabled(self):
        s = self.state()
        self.observe(s)
        before = copy.deepcopy(s.claims)
        self.assertEqual(self.observe(s)["action"], "observation_reused")
        self.assertEqual(s.claims, before)
        with self.assertRaises(ValueError): s.apply(None)

    def test_undecided_observation_survives_forbidden_repair(self):
        s = self.state()
        class Unknown:
            def verify(self, c): return {"kind": "atomic_source_proxy", "verdict": "insufficient", "original_claim": copy.deepcopy(c)}
        s.observe("current_request", self.target, fingerprint(s.claims[self.target]), Unknown())
        with self.assertRaises(ReplacementPreconditionError):
            s.propose("current_request", self.target, self.proposal(s, claim(text="Movie 1 has catalog year 2000.")), self.verifier)
        self.assertEqual(s.claims[self.target]["status"], "unresolved")


if __name__ == "__main__":
    unittest.main()
