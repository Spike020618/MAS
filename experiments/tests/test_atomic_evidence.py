import copy
import unittest

from experiments.paired_deliberation.atomic_evidence import (
    AtomicEvidencePipeline, reduce_relations, validate_atom, parser_request, relation_request)
from experiments.paired_deliberation.verified_state import VerificationError


def atom(scope="preference_state", polarity=True):
    return {"scope": scope, "subject": "seeker", "predicate": "wants", "object": "Drama",
            "polarity": polarity, "time": "current", "reason": "Original assertion scope."}


class AtomicTests(unittest.TestCase):
    sources = {"m0": {"role": "seeker", "text": "I want Drama."},
               "m1": {"role": "seeker", "text": "I do not want Drama."}}

    def relation(self, supports="m0", refutes="m1"):
        return {"relations": [{"source_id": supports, "relation": "supports", "quote": self.sources[supports]["text"]},
                              {"source_id": refutes, "relation": "refutes", "quote": self.sources[refutes]["text"]}],
                "reason": "Opposing same-scope sources.", "usage": {}}

    def test_positive_and_negative_state_both_abstain(self):
        for polarity, support, refute in [(True, "m0", "m1"), (False, "m1", "m0")]:
            self.assertEqual(reduce_relations(atom(polarity=polarity), self.sources,
                                             self.relation(support, refute))["verdict"], "insufficient")

    def test_opposing_utterance_does_not_erase_matching_utterance(self):
        r = reduce_relations(atom("utterance"), self.sources, self.relation())
        self.assertEqual(r["verdict"], "supported")
        self.assertFalse(r["evidence_conflict"])

    def test_missing_source_is_failure_not_implicit_irrelevance(self):
        result = self.relation()
        result["relations"].pop()
        with self.assertRaisesRegex(VerificationError, "coverage"):
            reduce_relations(atom(), self.sources, result)

    def test_unknown_duplicate_or_fabricated_sources_fail(self):
        for modification in ("unknown", "duplicate", "quote"):
            result = self.relation()
            if modification == "unknown":
                result["relations"][0]["source_id"] = "alien"
            elif modification == "duplicate":
                result["relations"][1] = copy.deepcopy(result["relations"][0])
            else:
                result["relations"][0]["quote"] = "Invented source text"
            with self.assertRaises(VerificationError):
                reduce_relations(atom(), self.sources, result)

    def test_unsupported_shape_does_not_weaken_a_compound(self):
        invalid = {**atom(), "scope": "unsupported"}
        with self.assertRaises(VerificationError):
            validate_atom(invalid)

    def test_pipeline_reads_opposite_uncited_source_and_charges_two_requests(self):
        class Client:
            def __init__(self):
                self.requests = []
            def complete(self, slot, request):
                self.requests.append(request)
                return {"report": atom() if request["kind"] == "atomic_interpretation" else AtomicTests().relation(),
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
        client = Client()
        result = AtomicEvidencePipeline(self.sources, client).verify({"text": "The seeker wants Drama.", "source_ids": ["m0"]}, "case")
        self.assertEqual(result["verdict"], "insufficient")
        self.assertEqual(len(client.requests), 2)
        self.assertEqual(set(result["source_ids"]), {"m0", "m1"})

    def test_unsupported_parser_result_never_triggers_certification(self):
        class Client:
            def __init__(self): self.calls = 0
            def complete(self, slot, request):
                self.calls += 1
                return {"report": {"scope": "unsupported", "subject": None, "predicate": None, "object": None,
                                   "polarity": None, "time": None, "reason": "Compound assertion."}, "usage": {}}
        client = Client()
        r = AtomicEvidencePipeline(self.sources, client).verify({"text": "A and B."}, "case")
        self.assertFalse(r["certification_attempted"])
        self.assertEqual(client.calls, 1)

    def test_parser_cannot_see_evidence_and_judge_cannot_see_parser_rationale(self):
        import json
        p = parser_request({"text": "The seeker wants Drama."}, self.sources)
        payload = json.loads(p["messages"][1]["content"])
        self.assertNotIn("available_dialogue", payload)
        self.assertNotIn("I do not want Drama.", p["messages"][1]["content"])
        judge = relation_request(atom(), self.sources)
        payload = json.loads(judge["messages"][1]["content"])
        self.assertNotIn("reason", payload["atom"])


if __name__ == "__main__":
    unittest.main()
