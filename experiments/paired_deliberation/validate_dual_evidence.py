"""Frozen 81-instance development validation + 18 paired controls + 4 regressions.

New instances are separate from earlier calibration; template families remain
related. Runtime inference never reads labels. No recommendation targets here.
"""
import argparse
import copy
import json
from collections import Counter
from pathlib import Path

from experiments.consensus_recommendation.common import ROOT, create, fingerprint, sha
from experiments.paired_deliberation.budget_v6 import AttemptClient
from experiments.paired_deliberation.client_v5 import RawDeepSeek, parse_raw
from experiments.paired_deliberation.conflict_verifier import DualEvidenceVerifier, request as dual_request
from experiments.paired_deliberation.durable_provider import DurableProvider
from experiments.paired_deliberation.pipeline_v5 import source_map
from experiments.paired_deliberation.requests_v5 import RequestsV5
from experiments.paired_deliberation.state_v6 import StateV6
from experiments.paired_deliberation.verified_state import EvidenceVerifier, VerificationError


def build():
    rows = []
    for n, (g, h, a, b) in enumerate([
        ("Crime", "Family", "Study Horizon", "Study Brook"),
        ("Historical Drama", "Action", "Study Comet", "Study Grove"),
        ("Adventure", "Romantic Comedy", "Study Lantern", "Study Field")]):
        categories = {
            "current_preference": ([f"For Saturday evening I am asking for {g}. Please exclude {h}."], [
                (f"For Saturday evening the seeker requests {g}.", "supported", "state"),
                (f"For Saturday evening the seeker requests {h}.", "refuted", "state"),
                (f"The seeker has enjoyed {g} for many years.", "insufficient", "state")]),
            "temporal_scope": ([f"I liked {g} when I was at school. Nowadays I dislike {g}."], [
                (f"The seeker liked {g} during their school years.", "supported", "state"),
                (f"Nowadays the seeker likes {g}.", "refuted", "state"),
                (f"Nowadays the seeker likes {h}.", "insufficient", "state")]),
            "subject_attribution": ([f"My cousin likes {g}. Personally, I dislike {g}."], [
                (f"The seeker's cousin likes {g}.", "supported", "state"),
                (f"The seeker likes {g}.", "refuted", "state"),
                (f"The seeker's cousin likes {h}.", "insufficient", "state")]),
            "negation": ([f"I have not seen {a}. I watched {b} last week."], [
                (f"The seeker has not seen {a}.", "supported", "state"),
                (f"The seeker has seen {a}.", "refuted", "state"),
                (f"The seeker enjoyed {b}.", "insufficient", "state")]),
            "partial_support": ([f"I enjoyed {a}. I have never seen {b}."], [
                (f"The seeker enjoyed {a}.", "supported", "state"),
                (f"The seeker has seen {b}.", "refuted", "state"),
                (f"The seeker enjoyed {a} and generally enjoys {g}.", "insufficient", "state")]),
            "mention_vs_experience": ([f"Could you describe {a}? I have not seen it."], [
                (f"The seeker asked for a description of {a}.", "supported", "utterance"),
                (f"The seeker has seen {a}.", "refuted", "state"),
                (f"The seeker liked {a}.", "insufficient", "state")]),
            "catalog_paraphrase": ([], [
                (f"According to the supplied catalog, {a} was released in {1987+n}.", "supported", "catalog"),
                (f"According to the supplied catalog, {a} was released in {2017+n}.", "refuted", "catalog"),
                (f"The supplied catalog says Orin directed {a}.", "insufficient", "catalog")]),
            "unresolved_conflict": ([f"At one point I said: I want {g} tonight.",
                f"At the very same point I also said: I do not want {g} tonight. "
                "These statements have equal standing. Neither overrides the other."], [
                (f"The seeker wants {g} tonight.", "insufficient", "unresolved_positive"),
                (f"The seeker does not want {g} tonight.", "insufficient", "unresolved_negative"),
                (f"The seeker said that they want {g} tonight.", "supported", "utterance_positive"),
                (f"The seeker said that they do not want {g} tonight.", "supported", "utterance_negative"),
                (f"All supplied statements agree that the seeker wants {g} tonight.", "refuted", "agreement_positive"),
                (f"All supplied statements agree that the seeker does not want {g} tonight.", "refuted", "agreement_negative")])}
        for category, (texts, statements) in categories.items():
            # Reverse displayed source order for one entire conflict context,
            # preserving source contents and explicit equal-standing scope.
            observations = [{"source_id": "message:%d" % i, "role": "seeker", "text": text}
                            for i, text in enumerate(texts)]
            if category == "unresolved_conflict" and n == 1:
                observations.reverse()
            evidence = {"case_id": "dual_constructed:%d:%s" % (n, category), "observations": observations,
                "previous_mentions_excluded_not_asserted_seen": [],
                "candidates_in_shared_retrieval_order": [{"source_id": "movie:%d" % i, "movie_id": str(i),
                    "title": a if i == 1 else "Independent Control %d" % i,
                    "year": 1987+n, "genres": [g]} for i in range(1, 6)]}
            for text, label, scope in statements:
                claim = {"claim_id": "c1", "text": text,
                    "source_ids": ["movie:1"] if category == "catalog_paraphrase" else [o["source_id"] for o in observations],
                    "candidate_ids": ["1"], "stance": "support", "depends_on": [],
                    "claim_type": "item_fact" if category == "catalog_paraphrase" else "explicit_preference"}
                sentinel = {"claim_id": "sentinel", "text": "Movie 5 has catalog year %d." % (1987+n),
                    "source_ids": ["movie:5"], "candidate_ids": ["5"], "stance": "support",
                    "depends_on": [], "claim_type": "item_fact"}
                inp = {"evidence": evidence, "reports": {"current_request": {"ranking": [str(i) for i in range(1, 6)],
                    "claims": [claim, sentinel], "updates": []}}}
                rows.append({"id": "dual_" + fingerprint(inp)[:16], "input": inp, "label": label,
                    "group": evidence["case_id"], "category": category, "scope": scope})
    return sorted(rows, key=lambda r: r["id"])


def prepare(output):
    output = Path(output)
    rows = build()
    inputs = [{"id": r["id"], **r["input"]} for r in rows]
    labels = [{key: r[key] for key in ("id", "label", "group", "category", "scope")} for r in rows]
    regression_root = ROOT / "experiments/results/paired_deliberation_d1_conflict_diag_r1"
    regressions = json.loads((regression_root / "inputs.json").read_text())
    create(output / "inputs.json", inputs)
    create(output / "labels.json", labels)
    create(output / "regression_inputs.json", regressions)
    files = [Path(__file__), Path(__file__).with_name("conflict_verifier.py"),
             *Path(__file__).parent.glob("*_v6.py")]
    files += [Path(__file__).with_name(n) for n in (
        "requests_v5.py", "verified_state.py", "durable_provider.py", "client_v5.py", "pipeline_v5.py")]
    files += [ROOT / "experiments/consensus_recommendation" / n for n in ("common.py", "protocol.py")]
    plan = {"version": "dual-evidence-validation-r1", "items": 81, "groups": 24,
        "old_comparison_ids": [r["id"] for r in rows if r["category"] == "unresolved_conflict"],
        "regression_items": 4, "maximum_requests": 103, "model": "deepseek-chat", "temperature": 0,
        "limits": {"max_calls": 103, "per_call_prompt_tokens": 8000, "per_call_completion_tokens": 1280,
                   "total_prompt_tokens": 300000, "total_completion_tokens": 131840},
        "gates": {"accuracy_min": .90, "recall_each_min": .80, "category_accuracy_min": .75,
                  "false_supported_max": 0, "invalid_max": 0, "sentinel_changed_max": 0,
                  "unresolved_decisive_max": 0, "regression_errors_max": 0},
        "inputs_hash": fingerprint(inputs), "labels_hash": fingerprint(labels),
        "regression_hash": fingerprint(regressions),
        "code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(files))},
        "runtime_labels_available": False, "pilot_allowed": False,
        "scope": "New constructed development instances, related template families. Paired legacy comparison on "
                 "18 prespecified conflict cases only. Four old probes are regression, not independent validation. "
                 "No natural-dialogue generalization or recommendation gain claim.",
        "stop_policy": "one fixed batch; no retry on unknown billing; no enlargement or prompt change based on scores"}
    plan["plan_hash"] = fingerprint(plan)
    create(output / "plan.json", plan)
    # Preserve the actual executable sources, not hashes alone.
    for path in sorted(set(files)):
        target = output / "source_snapshot" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != path.read_bytes():
            raise ValueError("source snapshot differs")
        target.write_bytes(path.read_bytes())
    return {"status": "prepared", "items": 81, "maximum_requests": 103, "plan_hash": plan["plan_hash"], "new_api_calls": 0}


def verify_plan(output):
    output = Path(output)
    plan = json.loads((output / "plan.json").read_text())
    if plan["plan_hash"] != fingerprint({k: v for k, v in plan.items() if k != "plan_hash"}):
        raise ValueError("plan changed")
    for name, expected in plan["code_sha256"].items():
        if sha(ROOT / name) != expected or sha(output / "source_snapshot" / name) != expected:
            raise ValueError("source or snapshot changed")
    for name, key in (("inputs.json", "inputs_hash"), ("regression_inputs.json", "regression_hash")):
        if fingerprint(json.loads((output / name).read_text())) != plan[key]:
            raise ValueError("input changed")
    return plan


def infer(row, arm, client):
    state = StateV6(row["reports"], row["evidence"])
    sources = source_map(row["evidence"])
    target, sentinel = "current_request:c1", "current_request:sentinel"
    def patch(key):
        return {"target_claim_id": key, "expected_hash": fingerprint(state.claims[key]), "replacement": None}
    state.apply("current_request", sentinel, patch(sentinel), EvidenceVerifier(sources))
    before = copy.deepcopy(state.claims[sentinel])
    def judge(payload):
        builder = dual_request if arm == "dual" else RequestsV5.verification
        record = client.complete(arm + "_" + row["id"], builder(payload))
        return {**record["report"], "usage": record["usage"]}
    verifier = DualEvidenceVerifier(sources, judge) if arm == "dual" else EvidenceVerifier(sources, judge)
    class Guard:
        def verify(self, claim):
            try:
                return verifier.verify(claim)
            except (ValueError, KeyError, TypeError) as exc:
                return {"verdict": "insufficient", "kind": "invalid_proxy", "verification_failed": True, "reason": str(exc)}
    event = state.apply("current_request", target, patch(target), Guard())
    return {"verification": event["old_verification"], "action": event["action"],
            "sentinel_preserved": before == state.claims[sentinel], "state": state.payload()}


def score(output, plan):
    output = Path(output)
    labels = json.loads((output / "labels.json").read_text())
    if fingerprint(labels) != plan["labels_hash"]:
        raise ValueError("labels changed")
    total, correct, confusion = Counter(), Counter(), Counter()
    invalid = false_supported = unresolved_decisive = sentinel_changed = transition_errors = 0
    pairs, conflict_omissions = [], []
    for row in labels:
        pred = json.loads((output / "predictions/dual" / (row["id"] + ".json")).read_text())
        v = pred["verification"]["verdict"]
        invalid += pred["verification"].get("verification_failed", False)
        false_supported += v == "supported" and row["label"] != "supported"
        sentinel_changed += not pred["sentinel_preserved"]
        transition_errors += pred["action"] != {"supported": "keep", "refuted": "retract", "insufficient": "defer"}[v]
        confusion[(row["label"], v)] += 1
        for axis in ("all", "label:" + row["label"], "category:" + row["category"], "scope:" + row["scope"]):
            total[axis] += 1
            correct[axis] += v == row["label"]
        if row["scope"].startswith("unresolved_"):
            unresolved_decisive += v != "insufficient"
            if not pred["verification"].get("evidence_conflict"):
                conflict_omissions.append(row["id"])
        if row["id"] in plan["old_comparison_ids"]:
            old = json.loads((output / "predictions/legacy" / (row["id"] + ".json")).read_text())
            pairs.append({"id": row["id"], "gold": row["label"], "legacy": old["verification"]["verdict"], "dual": v})
    regressions = json.loads((output / "regression_inputs.json").read_text())
    regression_errors = sum(json.loads((output / "predictions/regression" / (row["id"] + ".json")).read_text())[
        "verification"]["verdict"] != "insufficient" for row in regressions)
    rates = {axis: correct[axis] / count for axis, count in total.items()}
    g = plan["gates"]
    gates = {"accuracy": rates["all"] >= g["accuracy_min"],
        "class_recall": min(v for k, v in rates.items() if k.startswith("label:")) >= g["recall_each_min"],
        "category_accuracy": min(v for k, v in rates.items() if k.startswith("category:")) >= g["category_accuracy_min"],
        "no_false_support": false_supported <= g["false_supported_max"], "valid_outputs": invalid <= g["invalid_max"],
        "sentinel_preserved": sentinel_changed <= g["sentinel_changed_max"],
        "conflicts_stay_unresolved": unresolved_decisive <= g["unresolved_decisive_max"],
        "old_probe_regression": regression_errors <= g["regression_errors_max"], "state_consistency": transition_errors == 0}
    raw = [json.loads(p.read_text()) for p in (output / "slots").glob("*.raw.json")]
    bodies = [json.loads(r["raw"]["body"]) for r in raw]
    models = sorted({b.get("model") for b in bodies})
    gates["one_returned_model"] = len(models) == 1
    physical = {"responses": len(raw), "pending_intents": len(list((output / "slots").glob("*.intent.json"))),
        "prompt_tokens": sum(b["usage"]["prompt_tokens"] for b in bodies),
        "completion_tokens": sum(b["usage"]["completion_tokens"] for b in bodies), "models": models}
    gates["complete_batch"] = len(raw) == plan["maximum_requests"] and physical["pending_intents"] == 0
    result = {"pass": all(gates.values()), "gates": gates, "rates": rates,
        "false_supported": false_supported, "unresolved_decisive": unresolved_decisive,
        "invalid": invalid, "sentinel_changed": sentinel_changed, "state_transition_errors": transition_errors,
        "conflict_side_omissions": conflict_omissions, "regression_errors": regression_errors,
        "confusion": {a + "->" + b: count for (a, b), count in confusion.items()},
        "paired_conflict": {"items": len(pairs), "legacy_correct": sum(p["legacy"] == p["gold"] for p in pairs),
            "dual_correct": sum(p["dual"] == p["gold"] for p in pairs), "pairs": pairs},
        "physical": physical, "plan_hash": plan["plan_hash"], "pilot_allowed": False, "scope": plan["scope"]}
    create(output / "score.json", result)
    return result


def execute(output):
    output = Path(output)
    plan = verify_plan(output)
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    provider = DurableProvider(output / "slots", plan, RawDeepSeek(model=plan["model"]), parse_raw, True)
    client = AttemptClient(provider, plan["limits"])
    for row in json.loads((output / "inputs.json").read_text()):
        arms = ("dual", "legacy") if row["id"] in plan["old_comparison_ids"] else ("dual",)
        for arm in arms:
            create(output / "predictions" / arm / (row["id"] + ".json"), infer(row, arm, client))
    for row in json.loads((output / "regression_inputs.json").read_text()):
        create(output / "predictions/regression" / (row["id"] + ".json"), infer(row, "dual", client))
    return score(output, plan)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-authorized", action="store_true")
    args = parser.parse_args()
    result = execute(args.output) if args.execute_authorized else prepare(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
