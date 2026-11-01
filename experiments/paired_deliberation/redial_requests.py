"""Label-free ReDial request builders for E1 and E2."""
import copy

from experiments.consensus_recommendation.protocol import FOCUS, prompt


class RedialRequestBuilder:
    def __init__(self, evidence, max_tokens=1280):
        if not isinstance(evidence, dict) or "case_id" not in evidence:
            raise ValueError("frozen evidence bundle is required")
        self.evidence = copy.deepcopy(evidence)
        self.max_tokens = max_tokens

    def _envelope(self, messages, prior=None):
        return {"messages": messages, "evidence": copy.deepcopy(self.evidence),
                "prior": copy.deepcopy(prior), "max_tokens": self.max_tokens,
                "labels_available": False}

    def e1(self, case_id, method, stage, identity, prior):
        if case_id != self.evidence["case_id"]:
            raise ValueError("case/evidence mismatch")
        if stage == "initial":
            if method == "multi_role" and identity in FOCUS:
                instruction = FOCUS[identity]
            elif method == "central_all_evidence":
                instruction = "Use every available evidence source and make one centralized recommendation."
            else:
                instruction = "Independently make the best recommendation using all available evidence."
            messages = prompt(self.evidence, instruction)
        else:
            messages = prompt(
                self.evidence,
                "Review your own prior report for evidence misuse and improve it only when justified.",
                history=[prior],
            )
        return self._envelope(messages)

    def e2_review(self, case_id, request):
        if case_id != self.evidence["case_id"]:
            raise ValueError("case/evidence mismatch")
        agent = request["agent"]
        if request["mode"] == "self_review":
            instruction = "Review only your own report against the frozen evidence; revise only when justified."
        elif request["mode"] == "peer_review":
            instruction = "Review the visible peer reports and revise your own report when peer evidence justifies it."
        elif request["mode"] == "evidence_update":
            instruction = (
                "Issue exactly one accept/reject/unresolved update to one of your own qualified claims. "
                "Cite frozen evidence and then return your revised report. Your claim IDs are prefixed " + agent + ":."
            )
        else:
            raise ValueError("unsupported review mode")
        messages = prompt(self.evidence, instruction, reports=request["reports"])
        return self._envelope(messages, request["reports"])

    def aggregate(self, case_id, arm, state):
        if case_id != self.evidence["case_id"]:
            raise ValueError("case/evidence mismatch")
        instruction = (
            "Produce the final five-film recommendation from the supplied reports. "
            "Do not equate agreement with truth and preserve unresolved uncertainty. Arm identity is not supplied."
        )
        reports = ({"sample_%d" % index: report for index, report in enumerate(state)}
                   if isinstance(state, list) else state)
        if not isinstance(reports, dict):
            raise ValueError("aggregation reports must be a list or mapping")
        messages = prompt(self.evidence, instruction, reports=reports)
        envelope = self._envelope(messages, reports)
        envelope["aggregation_context"] = "paired reports"
        return envelope
