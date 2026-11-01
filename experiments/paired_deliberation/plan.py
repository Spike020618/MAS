"""Frozen method matrix for E1-E3 before any provider request."""
import copy
import hashlib
import json


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def build_plan(case_ids, model, budget, review_rounds=3):
    if not isinstance(case_ids, list) or not case_ids or len(case_ids) != len(set(case_ids)):
        raise ValueError("case IDs must be a nonempty unique list")
    if not isinstance(model, str) or not model:
        raise ValueError("model identity is required")
    if not isinstance(review_rounds, int) or review_rounds <= 0:
        raise ValueError("review rounds must be positive")
    methods = {
        "E1_single_once": {"family": "E1", "initial_generators": 1, "reviews": 0, "aggregations": 0},
        "E1_single_serial": {"family": "E1", "initial_generators": 1, "reviews": review_rounds, "aggregations": 0},
        "E1_same_agent_samples": {"family": "E1", "initial_generators": 3, "reviews": 0, "aggregations": 1},
        "E1_multi_role": {"family": "E1", "initial_generators": 3, "reviews": 0, "aggregations": 1},
        "E1_central_all_evidence": {"family": "E1", "initial_generators": 1, "reviews": review_rounds, "aggregations": 0},
        "E2_independent": {"family": "E2", "arm": "independent", "visibility": "frozen", "reviews": 0},
        "E2_self_review": {"family": "E2", "arm": "self_review", "visibility": "frozen", "reviews": review_rounds},
        "E2_peer_review": {"family": "E2", "arm": "peer_review", "visibility": "frozen", "reviews": review_rounds},
        "E2_evidence_update": {"family": "E2", "arm": "evidence_update", "visibility": "frozen", "reviews": review_rounds},
        "E2_peer_sequential": {"family": "E2_secondary", "arm": "peer_review", "visibility": "sequential", "reviews": review_rounds},
        "E2_update_sequential": {"family": "E2_secondary", "arm": "evidence_update", "visibility": "sequential", "reviews": review_rounds},
    }
    ablations = {
        "E3_random_or_all_routing": "replace fixed dispute routing only",
        "E3_peer_conclusion_only": "remove evidence content only",
        "E3_syntax_only": "remove semantic proxy evaluation only",
        "E3_unconstrained_regeneration": "remove gated state transition only",
        "E3_force_binary": "replace unresolved with forced accept/reject only",
    }
    plan = {
        "version": "paired-deliberation.execution-plan-v1",
        "case_ids": list(case_ids),
        "model": model,
        "budget": copy.deepcopy(budget),
        "review_rounds": review_rounds,
        "methods": methods,
        "ablations": ablations,
        "shared_initial_reports_for_e2": True,
        "labels_available_to_execution": False,
        "reserved_test_authorized": False,
        "provider_calls_authorized": False,
    }
    plan["plan_hash"] = fingerprint(plan)
    return plan


def authorize(plan, provider_calls=False, reserved_test=False):
    result = copy.deepcopy(plan)
    result.pop("plan_hash", None)
    result["provider_calls_authorized"] = bool(provider_calls)
    result["reserved_test_authorized"] = bool(reserved_test)
    result["plan_hash"] = fingerprint(result)
    return result
