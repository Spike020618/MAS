"""End-to-end E2 arm execution with shared reports and complete accounting."""
import copy

from experiments.paired_deliberation.ledger import ExperimentLedger
from experiments.paired_deliberation.runner import PairedRunner


def _ledger_add(ledger, stage, record, metadata):
    usage = record["usage"]
    ledger.add(stage, usage["prompt_tokens"], usage["completion_tokens"],
               usage["tool_calls"], usage["tool_units"], usage["retries"],
               record["elapsed_ms"], usage["actual_cost"], metadata)


def _assert_path_budget(summary, budget):
    for usage_key, budget_key in (("prompt_tokens", "total_prompt_tokens"),
                                  ("completion_tokens", "total_completion_tokens"),
                                  ("tool_calls", "total_tool_calls")):
        if summary[usage_key] > budget[budget_key]:
            raise ValueError("complete path exceeded " + budget_key)


def run_e1_method(case_id, method, roles, review_rounds, provider,
                  request_builder, aggregate_builder, budget):
    """Execute one E1 baseline without evaluation-label access."""
    allowed = {"single_once", "single_serial", "same_agent_samples",
               "multi_role", "central_all_evidence"}
    if method not in allowed:
        raise ValueError("unknown E1 method")
    if not isinstance(roles, list) or not roles:
        raise ValueError("roles are required")
    if not isinstance(review_rounds, int) or review_rounds <= 0:
        raise ValueError("review rounds must be positive")
    ledger = ExperimentLedger()
    models = set()
    reports = []

    def call(stage, identity, prior=None):
        slot = "%s_E1_%s_%s_%02d" % (
            case_id.replace(":", "_"), method, stage, len(reports)
        )
        payload = request_builder(case_id, method, stage, identity, copy.deepcopy(prior))
        if any(key in payload for key in ("gold", "labels", "relevant_movie_ids")):
            raise ValueError("E1 request contains evaluation labels")
        record = provider.complete(slot, payload)
        models.add(record["model"])
        _ledger_add(ledger, "initial_generation" if stage == "initial" else "review", record,
                    {"case_id": case_id, "method": method, "identity": identity})
        reports.append(copy.deepcopy(record["report"]))
        return record["report"]

    if method in {"same_agent_samples", "multi_role"}:
        identities = ["same_agent_%d" % i for i in range(3)] if method == "same_agent_samples" else roles
        current = [call("initial", identity) for identity in identities]
        payload = aggregate_builder(case_id, method, copy.deepcopy(current))
        if any(key in payload for key in ("gold", "labels", "relevant_movie_ids")):
            raise ValueError("E1 aggregation contains evaluation labels")
        slot = "%s_E1_%s_aggregate" % (case_id.replace(":", "_"), method)
        record = provider.complete(slot, payload)
        models.add(record["model"])
        _ledger_add(ledger, "aggregation", record,
                    {"case_id": case_id, "method": method, "identity": "shared"})
        decision = record["report"]
    else:
        identity = "central_all_evidence" if method == "central_all_evidence" else "single_agent"
        decision = call("initial", identity)
        if method in {"single_serial", "central_all_evidence"}:
            for _ in range(review_rounds):
                decision = call("review", identity, decision)

    if len(models) != 1:
        raise ValueError("model identity changed within E1 path")
    summary = ledger.summary()
    _assert_path_budget(summary, budget)
    return {"method": method, "decision": copy.deepcopy(decision),
            "reports": reports, "ledger": summary, "model": next(iter(models)),
            "evaluation_labels_available_to_execution": False}


def run_e2_arm(case_id, initial_records, observations, arm, schedule, budget,
               visibility, provider, request_builder, aggregate_builder, checker=None,
               path_name=None):
    """Execute one arm; evaluation labels are absent from every request builder."""
    if not isinstance(initial_records, dict) or set(initial_records) != set(observations):
        raise ValueError("initial records and observations must cover identical agents")
    ledger = ExperimentLedger()
    path_name = path_name or (arm + "_" + visibility)
    initial = {}
    models = set()
    for agent, record in initial_records.items():
        initial[agent] = copy.deepcopy(record["report"])
        models.add(record["model"])
        _ledger_add(ledger, "initial_generation", record,
                    {"case_id": case_id, "arm": arm, "agent": agent, "logical_shared": True})

    call_index = {"value": 0}

    def review(request):
        slot = "%s_%s_%02d" % (case_id.replace(":", "_"), path_name, call_index["value"])
        call_index["value"] += 1
        payload = request_builder(case_id, copy.deepcopy(request))
        if any(key in payload for key in ("gold", "labels", "relevant_movie_ids")):
            raise ValueError("execution request contains evaluation labels")
        record = provider.complete(slot, payload)
        models.add(record["model"])
        _ledger_add(ledger, "review", record,
                    {"case_id": case_id, "arm": arm, "slot": slot})
        return {"report": copy.deepcopy(record["report"]),
                "usage": {key: record["usage"][key]
                          for key in ("prompt_tokens", "completion_tokens", "tool_calls")}}

    aggregate_record = {"value": None}

    def aggregate(state):
        payload = aggregate_builder(case_id, arm, copy.deepcopy(state))
        if any(key in payload for key in ("gold", "labels", "relevant_movie_ids")):
            raise ValueError("aggregation request contains evaluation labels")
        slot = "%s_%s_aggregate" % (case_id.replace(":", "_"), path_name)
        record = provider.complete(slot, payload)
        models.add(record["model"])
        _ledger_add(ledger, "aggregation", record,
                    {"case_id": case_id, "arm": arm, "slot": slot})
        aggregate_record["value"] = record
        return copy.deepcopy(record["report"])

    result = PairedRunner(review, aggregate, checker).run(
        initial, observations, arm, schedule, budget, visibility
    )
    if len(models) != 1:
        raise ValueError("model identity changed within paired path")
    result["ledger"] = ledger.assert_complete_path(require_review=arm != "independent")
    _assert_path_budget(result["ledger"], budget)
    result["model"] = next(iter(models))
    result["aggregate_record"] = aggregate_record["value"]
    result["evaluation_labels_available_to_execution"] = False
    return result
