"""48 frozen constructed claims: natural entailment -> state, not recommendation."""
import argparse
import copy
import json
from collections import Counter
from pathlib import Path

from experiments.consensus_recommendation.common import ROOT, create, fingerprint, sha
from experiments.paired_deliberation.budget_v6 import AttemptClient
from experiments.paired_deliberation.client_v5 import RawDeepSeek, parse_raw
from experiments.paired_deliberation.durable_provider import DurableProvider
from experiments.paired_deliberation.pipeline_v5 import source_map
from experiments.paired_deliberation.requests_v5 import RequestsV5
from experiments.paired_deliberation.state_v6 import StateV6
from experiments.paired_deliberation.verified_state import EvidenceVerifier, VerificationError


def build():
    rows = []
    for index, (g, h, a, b) in enumerate([
        ("Mystery", "Western", "Screening Alpha", "Screening Beta"),
        ("Musical", "Science Fiction", "Screening Gamma", "Screening Delta")]):
        categories = {
            "current_preference": ([f"For this evening, please choose {g}; do not choose {h}."], [
                f"For this evening the seeker asks for {g}.", f"For this evening the seeker asks for {h}.",
                f"The seeker has liked {g} since childhood."]),
            "temporal_scope": ([f"Years ago I liked {g}. These days I do not like {g}."], [
                f"In the past the seeker liked {g}.", f"The seeker likes {g} these days.",
                f"The seeker likes {h} these days."]),
            "subject_attribution": ([f"My brother likes {g}, whereas I dislike {g}."], [
                f"The seeker's brother likes {g}.", f"The seeker likes {g}.",
                f"The seeker's brother likes {h}."]),
            "negation": ([f"I have never seen {a}. I saw {b} yesterday."], [
                f"The seeker has not seen {a}.", f"The seeker has seen {a}.",
                f"The seeker enjoyed {b}."]),
            "partial_support": ([f"I enjoyed {a}, and I have never watched {b}."], [
                f"The seeker enjoyed {a}.", f"The seeker has watched {b}.",
                f"The seeker enjoyed {a} and generally likes {g}."]),
            "conflicting_evidence": ([f"One statement says: I want {g} tonight.",
                f"Another statement at exactly the same time says: I do not want {g} tonight. "
                "Neither statement overrides the other; no resolution is provided."], [
                f"The evidence contains conflicting statements about wanting {g} tonight.",
                f"Both statements agree that the seeker wants {g} tonight.",
                f"The seeker's final resolved choice is {g} tonight."]),
            "mention_vs_experience": ([f"Please tell me about {a}; I have not watched it."], [
                f"The seeker requested information about {a}.", f"The seeker has watched {a}.",
                f"The seeker liked {a}."]),
            "catalog_paraphrase": ([], [f"The supplied catalog lists {a}'s release year as {2000 + index}.",
                f"The supplied catalog lists {a}'s release year as {2010 + index}.",
                f"The supplied catalog names Mira as the director of {a}."])}
        for category, (texts, claims) in categories.items():
            evidence = {"case_id": "constructed:%d:%s" % (index, category),
                "observations": [{"source_id": "message:%d" % i, "role": "seeker", "text": text}
                                 for i, text in enumerate(texts)],
                "previous_mentions_excluded_not_asserted_seen": [],
                "candidates_in_shared_retrieval_order": [{"source_id": "movie:%d" % i, "movie_id": str(i),
                    "title": a if i == 1 else "Control Film %d" % i, "year": 2000 + index,
                    "genres": [g]} for i in range(1, 6)]}
            for text, label in zip(claims, ("supported", "refuted", "insufficient")):
                claim = {"claim_id": "c1", "text": text,
                    "source_ids": ["movie:1"] if category == "catalog_paraphrase" else [obs["source_id"] for obs in evidence["observations"]],
                    "candidate_ids": ["1"], "stance": "support",
                    "claim_type": "item_fact" if category == "catalog_paraphrase" else "explicit_preference", "depends_on": []}
                sentinel = {"claim_id": "sentinel", "text": "Movie 5 has catalog year %s." % (2000 + index),
                    "source_ids": ["movie:5"], "candidate_ids": ["5"], "stance": "support",
                    "claim_type": "item_fact", "depends_on": []}
                inp = {"evidence": evidence, "reports": {"current_request": {
                    "ranking": [str(i) for i in range(1, 6)], "claims": [claim, sentinel], "updates": []}}}
                rows.append({"id": "controlled_" + fingerprint(inp)[:16], "input": inp,
                    "label": label, "category": category, "group": evidence["case_id"]})
    return sorted(rows, key=lambda row: row["id"])


def prepare(output):
    output = Path(output)
    rows = build()
    inputs = [{"id": row["id"], **row["input"]} for row in rows]
    labels = [{key: row[key] for key in ("id", "label", "category", "group")} for row in rows]
    create(output / "inputs.json", inputs)
    create(output / "labels.json", labels)
    code = [Path(__file__), *Path(__file__).parent.glob("*_v6.py"),
            Path(__file__).parent / "verified_state.py", Path(__file__).parent / "requests_v5.py",
            Path(__file__).parent / "durable_provider.py", Path(__file__).parent / "client_v5.py"]
    plan = {"version": "d1-controlled-natural-update-48-r1", "items": 48, "groups": 16,
        "model": "deepseek-chat", "temperature": 0, "maximum_requests": 48,
        "limits": {"max_calls": 48, "per_call_prompt_tokens": 8000, "per_call_completion_tokens": 1280,
                   "total_prompt_tokens": 128000, "total_completion_tokens": 61440},
        "gates": {"accuracy_min": .90, "recall_each_min": .80, "category_accuracy_min": .75,
                  "false_supported_max": 0, "invalid_max": 0, "sentinel_changed_max": 0},
        "inputs_hash": fingerprint(inputs), "labels_hash": fingerprint(labels),
        "code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(code))},
        "runtime_labels_available": False, "pilot_allowed": False,
        "scope": "constructed evidence-relative development check, new wording/values but related to prior template families; "
                 "no claim of independent natural-dialogue generalization or recommendation effect",
        "execution": "verify unrelated correct sentinel deterministically, then verify target once and apply null replacement; "
                     "does not test autonomous proposal quality, replacement quality, collaboration or scheduling gain"}
    plan["plan_hash"] = fingerprint(plan)
    create(output / "plan.json", plan)
    return plan


def score(output, plan):
    # Labels are accessed only after inference; runtime path never reads this file.
    output = Path(output)
    labels = json.loads((output / "labels.json").read_text())
    if fingerprint(labels) != plan["labels_hash"]:
        raise ValueError("labels changed")
    totals, correct, confusion = Counter(), Counter(), Counter()
    invalid, sentinel_changed, transition_correct = 0, 0, 0
    for row in labels:
        pred = json.loads((output / "predictions" / (row["id"] + ".json")).read_text())
        verdict = pred["verification"]["verdict"]
        invalid += pred["verification"].get("verification_failed", False)
        sentinel_changed += not pred["sentinel_preserved"]
        confusion[(row["label"], verdict)] += 1
        for axis in ("all", "label:" + row["label"], "category:" + row["category"]):
            totals[axis] += 1
            correct[axis] += verdict == row["label"]
        transition_correct += pred["action"] == {"supported": "keep", "refuted": "retract", "insufficient": "defer"}[row["label"]]
    rates = {axis: correct[axis] / total for axis, total in totals.items()}
    false_supported = sum(n for (gold, predicted), n in confusion.items() if predicted == "supported" and gold != "supported")
    g = plan["gates"]
    gates = {"accuracy": rates["all"] >= g["accuracy_min"],
        "class_recall": min(v for k, v in rates.items() if k.startswith("label:")) >= g["recall_each_min"],
        "category_accuracy": min(v for k, v in rates.items() if k.startswith("category:")) >= g["category_accuracy_min"],
        "false_supported": false_supported <= g["false_supported_max"], "valid_outputs": invalid <= g["invalid_max"],
        "sentinel_preserved": sentinel_changed <= g["sentinel_changed_max"]}
    physical = {"responses": 0, "prompt_tokens": 0, "completion_tokens": 0}
    for path in (output / "slots").glob("*.json"):
        if path.name.endswith((".raw.json", ".intent.json")):
            continue
        record = json.loads(path.read_text())["record"]
        physical["responses"] += 1
        for axis in ("prompt_tokens", "completion_tokens"):
            physical[axis] += record["usage"][axis]
    result = {"pass": all(gates.values()), "gates": gates, "rates": rates,
        "confusion": {a + "->" + b: n for (a, b), n in confusion.items()}, "false_supported": false_supported,
        "invalid": invalid, "sentinel_changed": sentinel_changed, "transition_correct": transition_correct,
        "items": len(labels), "groups": plan["groups"], "plan_hash": plan["plan_hash"], "physical": physical,
        "scope": plan["scope"], "pilot_allowed": False}
    create(output / "score.json", result)
    return result


def execute(output):
    output = Path(output)
    plan = json.loads((output / "plan.json").read_text())
    if plan["plan_hash"] != fingerprint({k: v for k, v in plan.items() if k != "plan_hash"}):
        raise ValueError("plan changed")
    for name, expected in plan["code_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("code changed")
    inputs = json.loads((output / "inputs.json").read_text())
    if fingerprint(inputs) != plan["inputs_hash"] or len(inputs) != plan["items"]:
        raise ValueError("input changed")
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    provider = DurableProvider(output / "slots", plan, RawDeepSeek(model=plan["model"]), parse_raw, True)
    client = AttemptClient(provider, plan["limits"])
    for row in inputs:
        state = StateV6(row["reports"], row["evidence"])
        sources = source_map(row["evidence"])
        target, sentinel = "current_request:c1", "current_request:sentinel"
        def patch(key):
            return {"target_claim_id": key, "expected_hash": fingerprint(state.claims[key]), "replacement": None}
        state.apply("current_request", sentinel, patch(sentinel), EvidenceVerifier(sources))
        before = copy.deepcopy(state.claims[sentinel])
        def judge(payload):
            record = client.complete(row["id"], RequestsV5.verification(payload))
            return {**record["report"], "usage": record["usage"]}
        class Judge:
            def verify(self, claim):
                try:
                    return EvidenceVerifier(sources, judge).verify(claim)
                except VerificationError as exc:
                    return {"verdict": "insufficient", "verification_failed": True, "reason": str(exc)}
        event = state.apply("current_request", target, patch(target), Judge())
        create(output / "predictions" / (row["id"] + ".json"), {
            "verification": event["old_verification"], "action": event["action"],
            "sentinel_preserved": before == state.claims[sentinel], "after": state.payload()})
    return score(output, plan)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-authorized", action="store_true")
    args = parser.parse_args()
    print(json.dumps(execute(args.output) if args.execute_authorized else prepare(args.output), ensure_ascii=False, indent=2))
