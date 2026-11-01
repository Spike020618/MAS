"""One fixed scope-aware revision check: 81 new instances + 21 regressions."""
import argparse
import copy
import json
import re
from pathlib import Path

from experiments.consensus_recommendation.common import ROOT, create, fingerprint, sha
from experiments.paired_deliberation.budget_v6 import AttemptClient
from experiments.paired_deliberation.client_v5 import RawDeepSeek, parse_raw
from experiments.paired_deliberation.durable_provider import DurableProvider
from experiments.paired_deliberation.pipeline_v5 import source_map
from experiments.paired_deliberation.scoped_verifier import ScopedEvidenceVerifier, request
from experiments.paired_deliberation.state_v6 import StateV6
from experiments.paired_deliberation.validate_dual_evidence import build as base_build, verify_plan, score as core_score
from experiments.paired_deliberation.verified_state import EvidenceVerifier

PRIOR = ROOT / "experiments/results/paired_deliberation_d1_dual_validation_r1"


def build():
    replacements = {"Historical Drama": "War", "Romantic Comedy": "Sport", "Adventure": "Fantasy",
        "Crime": "Thriller", "Family": "Animation", "Action": "Documentary",
        "Study Horizon": "Review Mariner", "Study Brook": "Review Pebble",
        "Study Comet": "Review Orchard", "Study Grove": "Review Cloud",
        "Study Lantern": "Review Summit", "Study Field": "Review Harbor",
        "Saturday": "Sunday", "cousin": "aunt", "school": "college",
        "dual_constructed:": "scoped_constructed:",
        "1987": "1974", "1988": "2004", "1989": "2022", "2017": "1984", "2018": "2014", "2019": "2024"}
    pattern = re.compile("|".join(re.escape(k) for k in sorted(replacements, key=len, reverse=True)))
    def transform(value):
        if isinstance(value, str):
            return pattern.sub(lambda m: replacements[m.group()], value)
        if isinstance(value, dict):
            return {k: transform(v) for k, v in value.items()}
        if isinstance(value, list):
            return [transform(v) for v in value]
        if type(value) is int:
            return {1987: 1974, 1988: 2004, 1989: 2022}.get(value, value)
        return value
    rows = []
    for row in base_build():
        item = transform(row)
        item["id"] = "scoped_" + fingerprint(item["input"])[:16]
        item["group"] = item["input"]["evidence"]["case_id"]
        rows.append(item)
    return sorted(rows, key=lambda r: r["id"])


def prepare(output):
    output = Path(output)
    original = json.loads((PRIOR / "plan.json").read_text())
    rows = build()
    inputs = [{"id": r["id"], **r["input"]} for r in rows]
    labels = [{k: r[k] for k in ("id", "label", "group", "category", "scope")} for r in rows]
    prior_inputs = {r["id"]: r for r in json.loads((PRIOR / "inputs.json").read_text())}
    extra_inputs, extra_labels = [], []
    for row in json.loads((PRIOR / "labels.json").read_text()):
        pred = json.loads((PRIOR / "predictions/dual" / (row["id"] + ".json")).read_text())
        if pred["verification"]["verdict"] != row["label"] or pred["verification"].get("verification_failed"):
            extra_inputs.append(prior_inputs[row["id"]])
            extra_labels.append(row)
    if len(extra_inputs) != 17:
        raise ValueError("prior failure set changed")
    regressions = json.loads((PRIOR / "regression_inputs.json").read_text())
    for name, value in (("inputs.json", inputs), ("labels.json", labels), ("regression_inputs.json", regressions),
                        ("extra_regression_inputs.json", extra_inputs), ("extra_regression_labels.json", extra_labels)):
        create(output / name, value)
    files = [ROOT / name for name in original["code_sha256"]]
    files += [Path(__file__), Path(__file__).with_name("scoped_verifier.py")]
    plan = {**original, "version": "scoped-dual-validation-r2-budget-only", "contract_version": "scoped_dual_v2",
        "maximum_requests": 102, "old_comparison_ids": [], "extra_regression_items": 17,
        "limits": {**original["limits"], "max_calls": 102, "total_prompt_tokens": 102 * 8000,
                   "total_completion_tokens": 130560},
        "inputs_hash": fingerprint(inputs), "labels_hash": fingerprint(labels), "regression_hash": fingerprint(regressions),
        "extra_regression_hash": fingerprint(extra_inputs), "extra_labels_hash": fingerprint(extra_labels),
        "code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(files))},
        "prior_score_sha256": sha(PRIOR / "score.json"),
        "scope": "Scope-aware v2 development check. 81 new value/wording instances in the same known template families, "
                 "not independent natural-dialogue generalization. 17 observed failures and 4 old conflict probes are "
                 "regression only; no new legacy baseline, no recommendation score. Release requires BOTH primary and regression gates."}
    plan.pop("plan_hash", None)
    plan["plan_hash"] = fingerprint(plan)
    create(output / "plan.json", plan)
    for path in sorted(set(files)):
        target = output / "source_snapshot" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != path.read_bytes():
            raise ValueError("snapshot differs")
        target.write_bytes(path.read_bytes())
    return {"status": "prepared", "items": 81, "regressions": 21, "maximum_requests": 102,
            "new_api_calls": 0, "plan_hash": plan["plan_hash"]}


def infer(row, client):
    state = StateV6(row["reports"], row["evidence"])
    sources = source_map(row["evidence"])
    target, sentinel = "current_request:c1", "current_request:sentinel"
    def patch(key):
        return {"target_claim_id": key, "expected_hash": fingerprint(state.claims[key]), "replacement": None}
    state.apply("current_request", sentinel, patch(sentinel), EvidenceVerifier(sources))
    before = copy.deepcopy(state.claims[sentinel])
    def judge(payload):
        record = client.complete("scoped_" + row["id"], request(payload))
        return {**record["report"], "usage": record["usage"]}
    verifier = ScopedEvidenceVerifier(sources, judge)
    class Guard:
        def verify(self, claim):
            try:
                return verifier.verify(claim)
            except (ValueError, KeyError, TypeError) as exc:
                return {"verdict": "insufficient", "kind": "invalid_proxy", "verification_failed": True, "reason": str(exc)}
    event = state.apply("current_request", target, patch(target), Guard())
    return {"verification": event["old_verification"], "action": event["action"],
            "sentinel_preserved": before == state.claims[sentinel], "state": state.payload()}


def execute(output):
    output = Path(output)
    plan = verify_plan(output)
    for name, key in (("extra_regression_inputs.json", "extra_regression_hash"), ("extra_regression_labels.json", "extra_labels_hash")):
        if fingerprint(json.loads((output / name).read_text())) != plan[key]:
            raise ValueError("regression artifact changed")
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    previous = output.parent / "paired_deliberation_d1_scoped_validation_r1/slots"
    client = AttemptClient(DurableProvider(output / "slots", plan, RawDeepSeek(model=plan["model"]), parse_raw, True,
        reuse_initial=previous if output.name.endswith("_r2") else None,
        reuse_all=output.name.endswith("_r2")), plan["limits"])
    for filename, folder in (("inputs.json", "dual"), ("regression_inputs.json", "regression"),
                             ("extra_regression_inputs.json", "extra_regression")):
        for row in json.loads((output / filename).read_text()):
            create(output / "predictions" / folder / (row["id"] + ".json"), infer(row, client))
    core = core_score(output, plan)
    errors = []
    extra_labels = json.loads((output / "extra_regression_labels.json").read_text())
    for row in extra_labels:
        pred = json.loads((output / "predictions/extra_regression" / (row["id"] + ".json")).read_text())
        if (pred["verification"]["verdict"] != row["label"] or pred["verification"].get("verification_failed")
                or not pred["sentinel_preserved"]):
            errors.append({"id": row["id"], "gold": row["label"], "prediction": pred["verification"]})
    result = {**core, "pass": core["pass"] and not errors,
              "extra_regression": {"items": len(extra_labels), "correct": len(extra_labels) - len(errors), "errors": errors},
              "contract_version": plan["contract_version"], "scope": plan["scope"]}
    create(output / "release_score.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-authorized", action="store_true")
    args = parser.parse_args()
    print(json.dumps(execute(args.output) if args.execute_authorized else prepare(args.output), ensure_ascii=False, indent=2))
