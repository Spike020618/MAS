"""Auditable paired execution for deliberation experiments.

This module contains no provider client, task labels, dispute router, or semantic
verifier. Those are explicit adapters. An evidence-update arm cannot run unless
its checker returns an auditable decision record.
"""
import copy
import hashlib
import json


ARMS = {"independent", "self_review", "peer_review", "evidence_update"}
VISIBILITY_MODES = {"frozen", "sequential"}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _nonnegative_int(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("invalid " + name)
    return value


def _validate_budget(budget):
    required = {
        "per_call_prompt_tokens",
        "per_call_completion_tokens",
        "total_prompt_tokens",
        "total_completion_tokens",
        "total_tool_calls",
    }
    if not isinstance(budget, dict) or set(budget) != required:
        raise ValueError("budget must contain the complete fixed accounting contract")
    checked = {key: _nonnegative_int(value, key) for key, value in budget.items()}
    if checked["per_call_prompt_tokens"] <= 0 or checked["per_call_completion_tokens"] <= 0:
        raise ValueError("per-call token caps must be positive")
    return checked


def _usage(response):
    usage = response.get("usage")
    if not isinstance(usage, dict):
        raise ValueError("review adapter must report usage")
    return {
        "prompt_tokens": _nonnegative_int(usage.get("prompt_tokens"), "prompt_tokens"),
        "completion_tokens": _nonnegative_int(usage.get("completion_tokens"), "completion_tokens"),
        "tool_calls": _nonnegative_int(usage.get("tool_calls"), "tool_calls"),
    }


def _validate_check(check):
    if not isinstance(check, dict):
        raise ValueError("checker must return an auditable record")
    required = {"decision", "target_claim_ids", "evidence_refs", "rationale"}
    if set(check) != required:
        raise ValueError("checker record is incomplete or has undeclared fields")
    if check["decision"] not in {"accept", "reject", "unresolved"}:
        raise ValueError("checker must return an explicit update decision")
    if not all(isinstance(check[key], list) for key in ("target_claim_ids", "evidence_refs")):
        raise ValueError("checker targets and evidence references must be lists")
    if not isinstance(check["rationale"], str) or not check["rationale"].strip():
        raise ValueError("checker rationale is required")
    if check["decision"] == "accept" and (not check["target_claim_ids"] or not check["evidence_refs"]):
        raise ValueError("accepted updates require targets and evidence")
    return copy.deepcopy(check)


class PairedRunner:
    def __init__(self, review, aggregate, admissible=None):
        self.review = review
        self.aggregate = aggregate
        self.admissible = admissible

    def run(self, initial, observations, arm, schedule, budget, visibility="frozen"):
        """Run one arm from a shared initial-report snapshot.

        ``frozen`` makes every reviewer inspect the same initial snapshot and is
        the primary causal comparison. ``sequential`` exposes accepted earlier
        updates and is a separate, order-sensitive treatment. Initial-generation
        and aggregation costs must be added by the experiment-level ledger.
        """
        if arm not in ARMS:
            raise ValueError("unknown arm")
        if visibility not in VISIBILITY_MODES:
            raise ValueError("unknown visibility mode")
        limits = _validate_budget(budget)
        if not isinstance(initial, dict) or not initial:
            raise ValueError("invalid initial reports")
        if not isinstance(observations, dict) or set(initial) - set(observations):
            raise ValueError("missing agent observations")
        if not isinstance(schedule, list) or any(agent not in initial for agent in schedule):
            raise ValueError("scheduled unknown agent")
        if visibility == "frozen" and len(schedule) != len(set(schedule)):
            raise ValueError("frozen mode permits at most one proposal per agent")
        if arm == "evidence_update" and self.admissible is None:
            raise NotImplementedError("an actual evidence/update checker is required")

        snapshot = copy.deepcopy(initial)
        initial_hash = digest(snapshot)
        state = copy.deepcopy(snapshot)
        events = []
        totals = {"prompt_tokens": 0, "completion_tokens": 0, "tool_calls": 0}

        if arm != "independent":
            for step, agent in enumerate(schedule):
                source = snapshot if visibility == "frozen" else state
                visible = {agent: source[agent]} if arm == "self_review" else source
                request = {
                    "step": step,
                    "agent": agent,
                    "mode": arm,
                    "visibility": visibility,
                    "reports": copy.deepcopy(visible),
                    "observation": copy.deepcopy(observations[agent]),
                    "budget": copy.deepcopy(limits),
                }
                response = self.review(copy.deepcopy(request))
                if not isinstance(response, dict) or "report" not in response:
                    raise ValueError("review adapter returned no report")
                usage = _usage(response)
                if usage["prompt_tokens"] > limits["per_call_prompt_tokens"]:
                    raise ValueError("per-call prompt-token budget exceeded")
                if usage["completion_tokens"] > limits["per_call_completion_tokens"]:
                    raise ValueError("per-call completion-token budget exceeded")
                for key in totals:
                    totals[key] += usage[key]
                if totals["prompt_tokens"] > limits["total_prompt_tokens"]:
                    raise ValueError("total prompt-token budget exceeded")
                if totals["completion_tokens"] > limits["total_completion_tokens"]:
                    raise ValueError("total completion-token budget exceeded")
                if totals["tool_calls"] > limits["total_tool_calls"]:
                    raise ValueError("total tool-call budget exceeded")

                proposal = copy.deepcopy(response["report"])
                check = None
                if arm == "evidence_update":
                    check = _validate_check(
                        self.admissible(
                            agent,
                            copy.deepcopy(source[agent]),
                            proposal,
                            copy.deepcopy(observations[agent]),
                            copy.deepcopy(source),
                        )
                    )
                    decision = check["decision"]
                else:
                    decision = "accept"
                before = digest(state[agent])
                if decision == "accept":
                    state[agent] = proposal
                events.append(
                    {
                        "step": step,
                        "agent": agent,
                        "visibility": visibility,
                        "visible_report_hash": digest(visible),
                        "visible_agents": sorted(visible),
                        "proposal": proposal,
                        "check": check,
                        "decision": decision,
                        "before": before,
                        "after": digest(state[agent]),
                        "usage": usage,
                    }
                )

        if digest(initial) != initial_hash:
            raise ValueError("initial reports were mutated")
        return {
            "arm": arm,
            "visibility": visibility,
            "initial_hash": initial_hash,
            "initial_reports": snapshot,
            "final_reports": state,
            "events": events,
            "decision": self.aggregate(copy.deepcopy(state)),
            "review_calls": len(events),
            "scheduled_review_calls": 0 if arm == "independent" else len(schedule),
            "usage": totals,
            "budget": limits,
            "scope": "paired execution only; initial generation, aggregation, adapters, and labels are external",
        }
