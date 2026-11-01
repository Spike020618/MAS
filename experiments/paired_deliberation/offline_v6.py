"""Offline release gate; explicitly not a semantic or recommendation evaluation."""
import argparse
import copy
import io
import json
import unittest
from pathlib import Path

from experiments.consensus_recommendation.common import create, fingerprint
from experiments.paired_deliberation.contracts_v6 import OWNERS, initial_request, review_request
from experiments.paired_deliberation.pipeline_v5 import source_map
from experiments.paired_deliberation.preflight_v6 import verify_plan
from experiments.paired_deliberation.state_v6 import StateV6
from experiments.paired_deliberation.verified_state import EvidenceVerifier


def controlled_state_cases(cases):
    """Year mutations test execution only, using frozen source-derived labels."""
    inputs, labels, predictions = [], [], []
    for evidence in cases:
        facts = [fact for fact in evidence["candidates_in_shared_retrieval_order"] if type(fact.get("year")) is int][:6]
        for fact in facts:
            for mode in ("true", "false", "missing"):
                current = copy.deepcopy(evidence)
                if mode == "missing":
                    for row in current["candidates_in_shared_retrieval_order"]:
                        if row["movie_id"] == fact["movie_id"]:
                            row["year"] = None
                year = fact["year"] + int(mode == "false")
                claim = {"claim_id": "c1", "text": "Movie %s has catalog year %s." % (fact["movie_id"], year),
                    "source_ids": [fact["source_id"]], "candidate_ids": [fact["movie_id"]],
                    "stance": "support", "claim_type": "item_fact", "depends_on": []}
                reports = {OWNERS[0]: {"ranking": [m["movie_id"] for m in current[
                    "candidates_in_shared_retrieval_order"][:5]], "claims": [claim], "updates": []}}
                target = OWNERS[0] + ":c1"
                item_id = fingerprint({"case": evidence["case_id"], "movie": fact["movie_id"], "mode": mode})[:16]
                state = StateV6(reports, current)
                event = state.apply(OWNERS[0], target, {"target_claim_id": target,
                    "expected_hash": fingerprint(state.claims[target]), "replacement": None},
                    EvidenceVerifier(source_map(current)))
                expected = {"true": "keep", "false": "retract", "missing": "defer"}[mode]
                inputs.append({"id": item_id, "evidence": current, "reports": reports})
                labels.append({"id": item_id, "action": expected, "group": evidence["case_id"] + ":" + fact["movie_id"]})
                predictions.append({"id": item_id, "action": event["action"], "status": event["after"]["status"]})
    return inputs, labels, predictions


def run(output):
    output = Path(output)
    plan = verify_plan(output)
    stream = io.StringIO()
    modules = ("experiments.tests.test_design_v6", "experiments.tests.test_comparison_v51",
               "experiments.tests.test_verified_updates_v5")
    suite = unittest.TestLoader().loadTestsFromNames(modules)
    tests = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    cases = [json.loads((output / "inputs" / (cid.replace(":", "_") + ".json")).read_text())
             for cid in plan["case_ids"]]
    inputs, labels, predictions = controlled_state_cases(cases)
    create(output / "controlled_execution/inputs.json", inputs)
    create(output / "controlled_execution/labels.json", labels)
    create(output / "controlled_execution/predictions.json", predictions)
    correct = sum(gold["action"] == pred["action"] for gold, pred in zip(labels, predictions))
    replay = json.loads((output / "historical_contract_replay.json").read_text())
    checks = {"targeted_tests": tests.wasSuccessful(), "historical_16_contract_1_semantic": (
        replay["rejected"] == 17 and replay["counts"].get("semantic_precondition") == 1),
        "controlled_execution": bool(labels) and correct == len(labels)}
    result = {"pass": all(checks.values()), "checks": checks, "tests_run": tests.testsRun,
        "test_log": stream.getvalue(), "plan_hash": plan["plan_hash"],
        "controlled_execution": {"cases": len(labels), "correct": correct,
            "context_groups": len({row["group"] for row in labels}),
            "scope": "deterministic catalog state execution only; no natural semantics or recommendation effects"},
        "new_api_calls": 0, "pilot_allowed": False}
    create(output / "offline_acceptance.json", result)
    return {key: value for key, value in result.items() if key != "test_log"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output), ensure_ascii=False, indent=2))
