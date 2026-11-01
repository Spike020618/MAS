"""Five D1 conditions with on-demand verification; no evaluation labels."""
import copy

from experiments.consensus_recommendation.common import fingerprint
from experiments.paired_deliberation.budget_v6 import BudgetExceeded
from experiments.paired_deliberation.contracts_v6 import (
    METHODS, OWNERS, check_report, check_ranking, initial_request, review_request, ranking_request)
from experiments.paired_deliberation.coordination_v6 import Coordinator
from experiments.paired_deliberation.pipeline_v5 import source_map
from experiments.paired_deliberation.requests_v5 import RequestsV5
from experiments.paired_deliberation.state_v6 import StateV6, FACT_TYPES, ReplacementPreconditionError
from experiments.paired_deliberation.verified_state import EvidenceVerifier, VerificationError


def shared_initial(evidence, client, prefix):
    initial, calls = {}, []
    for owner in OWNERS:
        request = initial_request(evidence, owner)
        slot = prefix + "_initial_" + owner
        record = client.complete(slot, request)
        check_report(record["report"], evidence)
        initial[owner] = record["report"]
        calls.append({"slot": slot, "request": request, "record": record})
    StateV6(initial, evidence)  # Cross-report dependency integrity before any comparison.
    body = {"initial": initial, "calls": calls, "evidence_hash": fingerprint(evidence)}
    return {**body, "integrity": fingerprint(body)}


def check_shared(shared, evidence):
    if (shared.get("integrity") != fingerprint({k: v for k, v in shared.items() if k != "integrity"})
            or shared.get("evidence_hash") != fingerprint(evidence)
            or set(shared["initial"]) != set(OWNERS)):
        raise ValueError("shared initial integrity mismatch")


def run_method(evidence, method, client, prefix, shared=None, max_rounds=3):
    if method not in METHODS:
        raise ValueError("unknown D1 method")
    # Explicit allowlist prevents accidentally passing evaluation metadata to any LLM.
    if set(evidence) != {"case_id", "observations", "previous_mentions_excluded_not_asserted_seen",
                         "candidates_in_shared_retrieval_order"}:
        raise ValueError("unexpected evidence root fields")
    if shared is not None:
        check_shared(shared, evidence)
        initial = copy.deepcopy(shared["initial"])
        for call in shared["calls"]:
            client.account_shared(**call)
    elif method == "central":
        req = initial_request(evidence, "central")
        initial = {"central": client.complete(prefix + "_initial_central", req)["report"]}
    else:
        initial = shared_initial(evidence, client, prefix)["initial"]
    state = StateV6(initial, evidence)
    mode = "adaptive" if method == "verified_adaptive" else "fixed"
    coordinator = Coordinator(mode, initial, max_rounds)
    cache, feedback, errors, history, tool_events = {}, {}, [], [], []
    verify_index = [0]

    def verify(claim):
        key = fingerprint(claim)
        if key in cache:
            return copy.deepcopy(cache[key])
        if len(tool_events) >= client.limits.get("max_evidence_checks", 6):
            raise BudgetExceeded("evidence check budget exhausted")
        tool_event = {"claim_hash": key, "kind": "evidence_check", "status": "attempted"}
        tool_events.append(tool_event)
        def judge(payload):
            slot = prefix + "_verify_%03d" % verify_index[0]
            verify_index[0] += 1
            response = client.complete(slot, RequestsV5.verification(payload))
            return {**response["report"], "usage": response["usage"]}
        try:
            result = EvidenceVerifier(source_map(evidence), judge).verify(claim)
        except VerificationError as exc:
            result = {"verdict": "insufficient", "kind": "invalid_proxy", "reason": str(exc),
                      "verification_failed": True, "usage": {}}
        tool_event.update(status="completed", verifier_kind=result["kind"],
                          semantic_proxy=result["kind"] in {"model_proxy", "invalid_proxy"})
        cache[key] = result
        return copy.deepcopy(result)

    class Verifier:
        def verify(self, claim):
            return verify(claim)

    stop = "independent" if method == "independent" else "round_limit"
    if method != "independent":
        for index in range(max_rounds):
            # Central: select + old verify + whole review; peer: old verify + review.
            # Scoped: old verify + review + optional replacement verify. Reserve final too.
            worst = 3 if method != "peer" else 2
            if not client.action_fits(worst):
                stop = "budget_reserved_for_final"
                break
            if method == "central":
                eligible = sorted(key for key, row in state.claims.items()
                                  if key not in coordinator.processed and row["own_status"] != "retracted"
                                  and row["claim"]["claim_type"] in FACT_TYPES)
                if not eligible:
                    stop = "no_unprocessed_factual_targets"
                    break
                request = RequestsV5.envelope("select_v6", {"evidence": evidence, "state": state.payload(),
                    "allowed_targets": eligible, "history": history, "schema": {"target_claim_id": None}},
                    "Choose which claim to examine next using the evidence, or stop with null. "
                    "Return only JSON with target_claim_id, one allowed ID or null. Do not use external knowledge.")
                selection = client.complete(prefix + "_select_%03d" % index, request)["report"]
                if not isinstance(selection, dict) or set(selection) != {"target_claim_id"}:
                    raise ValueError("invalid central target selection")
                target = selection["target_claim_id"]
                if target is None:
                    stop = "controller_stop"
                    break
                if target not in eligible:
                    raise ValueError("central chose unavailable target")
                coordinator.processed.add(target)
                coordinator.trace.append({"target": target, "mode": "central_selected"})
            else:
                target = coordinator.next(state)
                if target is None:
                    stop = "no_unprocessed_factual_targets"
                    break
            before = copy.deepcopy(state.claims[target])
            check = verify(before["claim"])
            feedback[target] = {"claim_hash": fingerprint(before["claim"]), "verification": check}
            owner = before["owner"]
            reviewer = "central" if method == "central" else OWNERS[(OWNERS.index(owner) + 1) % len(OWNERS)]
            whole = method in {"central", "peer"}
            request = review_request(evidence, state, target, reviewer, whole, history if method == "central" else None)
            # Same selected-target feedback access; no free global pre-verification.
            import json
            payload = json.loads(request["messages"][1]["content"])
            payload["verification_feedback"] = copy.deepcopy(feedback)
            request["messages"][1]["content"] = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            proposal = client.complete(prefix + "_review_%03d" % index, request)["report"]
            try:
                if whole:
                    state.replace_owner(owner, proposal)
                else:
                    state.apply(reviewer, target, proposal, Verifier())
            except (ValueError, KeyError, TypeError) as exc:
                errors.append({"target": target, "error": str(exc), "proposal": copy.deepcopy(proposal),
                               "kind": "semantic_precondition" if isinstance(exc, ReplacementPreconditionError) else "contract"})
            history.append({"target": target, "feedback": copy.deepcopy(feedback[target]),
                            "proposal": copy.deepcopy(proposal), "state_hash": fingerprint(state.claims)})
    request = ranking_request(evidence, state)
    # Feedback is bound by exact claim hash. Never attach old verdicts to rewrites.
    import json
    payload = json.loads(request["messages"][1]["content"])
    payload["verification_feedback"] = {target: row for target, row in feedback.items()
        if target in state.claims and row["claim_hash"] == fingerprint(state.claims[target]["claim"])}
    request["messages"][1]["content"] = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    final = client.complete(prefix + "_aggregate", request)["report"]
    check_ranking(final, evidence)
    return {"method": method, "final": final, "state": state.payload(), "events": state.events,
            "schedule": coordinator.trace, "stop": stop, "rejected": errors,
            "verification_cache": cache, "tool_events": tool_events, "usage": client.summary(), "evidence_hash": fingerprint(evidence),
            "shared_initial": shared is not None, "evaluation_labels_available": False}


def run_suite(evidence, client_factory, shared=None, max_rounds=3):
    results = {}
    for method in METHODS:
        client = client_factory(method)
        prefix = evidence["case_id"].replace(":", "_") + "_" + method
        try:
            result = run_method(evidence, method, client, prefix, shared, max_rounds)
            results[method] = {"status": "completed", "result": result}
        except (ValueError, KeyError, TypeError, RuntimeError) as exc:
            results[method] = {"status": "failed", "error": str(exc), "usage": client.summary(),
                               "fallback": {"ranking": [row["movie_id"] for row in
                                  evidence["candidates_in_shared_retrieval_order"][:5]]}}
            if isinstance(exc, RuntimeError):
                # A pending request may already be billed; no automatic repeat or further dispatch.
                break
    return results
