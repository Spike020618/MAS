"""Fixed v3 check with runtime-produced scopes; never supplies oracle scopes."""
import argparse
import copy
import json
from collections import Counter
from pathlib import Path

from experiments.consensus_recommendation.common import ROOT, create, fingerprint, sha
from experiments.paired_deliberation.atomic_evidence import AtomicEvidencePipeline
from experiments.paired_deliberation.budget_v6 import AttemptClient
from experiments.paired_deliberation.client_v5 import RawDeepSeek, parse_raw
from experiments.paired_deliberation.durable_provider import DurableProvider
from experiments.paired_deliberation.pipeline_v5 import source_map
from experiments.paired_deliberation.state_v6 import StateV6
from experiments.paired_deliberation.verified_state import EvidenceVerifier


def expected_atom(scope, predicate, obj, polarity, time):
    return {"scope": scope, "subject": "seeker", "predicate": predicate, "object": obj,
            "polarity": polarity, "time": time}


def build():
    rows = []
    def add(evidence, text, expected, verdict, relations, category, cite=None):
        claim = {"claim_id": "c1", "text": text, "source_ids": cite or [o["source_id"] for o in evidence["observations"]],
                 "candidate_ids": ["1"], "stance": "support", "claim_type": "explicit_preference", "depends_on": []}
        sentinel = {"claim_id": "sentinel", "text": "Movie 5 has catalog year 2000.", "source_ids": ["movie:5"],
                    "candidate_ids": ["5"], "stance": "support", "claim_type": "item_fact", "depends_on": []}
        inp = {"evidence": copy.deepcopy(evidence), "reports": {"current_request": {
            "ranking": [str(i) for i in range(1, 6)], "claims": [claim, sentinel], "updates": []}}}
        rows.append({"id": "atomic_" + fingerprint(inp)[:16], "input": inp, "expected_atom": expected,
                     "label": verdict, "relations": relations, "category": category,
                     "group": evidence["case_id"], "supported_scope": expected["scope"] != "unsupported"})
    for index, (g, h) in enumerate((("Crime", "Romance"), ("War", "Science Fiction"),
                                    ("Historical Drama", "Family"), ("Mystery", "Action"))):
        def evidence(suffix, observations):
            return {"case_id": "atomic_context:%d:%s" % (index, suffix), "observations": observations,
                "previous_mentions_excluded_not_asserted_seen": [],
                "candidates_in_shared_retrieval_order": [{"source_id": "movie:%d" % i, "movie_id": str(i),
                    "title": "Atomic Control %d" % i, "year": 2000, "genres": []} for i in range(1, 6)]}
        direct = evidence("direct", [{"source_id": "message:0", "role": "seeker",
            "text": f"For the current session I want {g}. I do not want {h} for this session."}])
        for obj, verdict, relation in ((g, "supported", "supports"), (h, "refuted", "refutes"),
                                      ("Film noir", "insufficient", "irrelevant")):
            add(direct, f"For the current session the seeker wants {obj}.",
                expected_atom("preference_state", "wants", obj, True, "current"), verdict,
                {"message:0": relation}, "direct_preference")
        pos, neg = ("message:0", "message:1") if index % 2 == 0 else ("message:1", "message:0")
        conflict = evidence("conflict", [
            {"source_id": pos, "role": "seeker", "text": f"At the same moment I said: I want {g} tonight."},
            {"source_id": neg, "role": "seeker", "text": f"At that identical moment I also said: I do not want {g} tonight. "
             "Both statements have equal standing; neither supersedes the other."}])
        for scope in ("preference_state", "utterance"):
            for polarity in (True, False):
                content = ("wants " if polarity else "does not want ") + g + " tonight."
                text = ("The seeker " + content if scope == "preference_state" else
                        "In the supplied dialogue the seeker said they " + ("want " if polarity else "do not want ") + g + " tonight.")
                add(conflict, text, expected_atom(scope, "wants", g, polarity, "current"),
                    "insufficient" if scope == "preference_state" else "supported",
                    {pos: "supports" if polarity else "refutes", neg: "refutes" if polarity else "supports"},
                    "conflict_" + scope, cite=[pos if polarity else neg])
        history = evidence("history", [{"source_id": "message:0", "role": "seeker",
            "text": f"I liked {g} in the past. Currently I do not like {g}."}])
        for text, obj, time, verdict, relation in (
            (f"In the past the seeker liked {g}.", g, "past", "supported", "supports"),
            (f"The seeker currently likes {g}.", g, "current", "refuted", "refutes"),
            (f"The seeker currently likes {h}.", h, "current", "insufficient", "irrelevant")):
            add(history, text, expected_atom("preference_state", "likes", obj, True, time),
                verdict, {"message:0": relation}, "temporal_preference")
        unsupported = {"scope": "unsupported", "subject": None, "predicate": None, "object": None,
                       "polarity": None, "time": None}
        add(direct, f"The seeker wants {g} and wants {h} for the current session.", unsupported,
            "insufficient", {}, "unsupported_compound")
        add(conflict, f"All supplied statements agree that the seeker wants {g} tonight.", unsupported,
            "insufficient", {}, "unsupported_universal")
    return sorted(rows, key=lambda row: row["id"])


def regressions():
    root = ROOT / "experiments/results"
    inputs = json.loads((root / "paired_deliberation_d1_conflict_diag_r1/inputs.json").read_text())
    extra = json.loads((root / "paired_deliberation_d1_scoped_validation_r2/inputs.json").read_text())
    inputs += [r for r in extra if r["id"] == "scoped_58dfbabdc3b500f2"]
    rows = []
    for row in inputs:
        text = row["reports"]["current_request"]["claims"][0]["text"]
        negative = "does not want" in text
        obj = text.split("does not want " if negative else "wants ", 1)[1].removesuffix(" tonight.")
        rows.append({"id": row["id"], "input": {k: row[k] for k in ("evidence", "reports")},
            "expected_atom": expected_atom("preference_state", "wants", obj, not negative, "current"),
            "label": "insufficient", "relations": {"message:0": "refutes" if negative else "supports",
                                                     "message:1": "supports" if negative else "refutes"},
            "category": "old_conflict_regression", "group": row["evidence"]["case_id"], "supported_scope": True})
    if len(rows) != 5:
        raise ValueError("expected five frozen conflict regressions")
    prior = root / "paired_deliberation_d1_atomic_validation_r1"
    failures = {row["id"] for row in json.loads((prior / "release_score.json").read_text())["failures"]}
    old_inputs = {r["id"]: r for name in ("inputs.json", "regression_inputs.json")
                  for r in json.loads((prior / name).read_text())}
    known = {row["id"] for row in rows}
    for label in json.loads((prior / "labels.json").read_text()):
        if label["id"] in failures and label["id"] not in known:
            rows.append({**label, "input": {k: old_inputs[label["id"]][k] for k in ("evidence", "reports")},
                         "category": "old_atomic_failure_regression"})
            known.add(label["id"])
    return rows


def prepare(output):
    output = Path(output)
    rows, old = build(), regressions()
    inputs = [{"id": row["id"], **row["input"]} for row in rows]
    old_inputs = [{"id": row["id"], **row["input"]} for row in old]
    labels = [{k: v for k, v in row.items() if k != "input"} for row in rows + old]
    for name, data in (("inputs.json", inputs), ("regression_inputs.json", old_inputs), ("labels.json", labels)):
        create(output / name, data)
    names = ("atomic_evidence.py", "validate_atomic_evidence.py", "budget_v6.py", "state_v6.py", "contracts_v6.py",
             "pipeline_v5.py", "requests_v5.py", "verified_state.py", "durable_provider.py", "client_v5.py")
    files = [Path(__file__).with_name(n) for n in names] + [ROOT / "experiments/consensus_recommendation" / n for n in ("common.py", "protocol.py")]
    maximum_requests = 2 * (len(rows) + len(old))
    plan = {"version": "atomic-evidence-v3-validation-r2-parser-isolation", "contract_version": "atomic_v3", "model": "deepseek-chat",
        "new_items": 48, "atomic_primary_items": 40, "unsupported_stress_items": 8, "regressions": len(old),
        "maximum_requests": maximum_requests, "expected_requests_if_scopes_correct": maximum_requests - 8,
        "limits": {"max_calls": maximum_requests, "per_call_prompt_tokens": 8000, "per_call_completion_tokens": 4096,
                   "total_prompt_tokens": maximum_requests * 8000, "total_completion_tokens": maximum_requests * 4096},
        "inputs_hash": fingerprint(inputs), "regression_hash": fingerprint(old_inputs), "labels_hash": fingerprint(labels),
        "code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in files},
        "gates": {"primary_accuracy_min": .90, "recall_each_min": .80, "false_supported_max": 0,
                  "interpretation_errors_max": 0, "invalid_max": 0, "unresolved_decisive_max": 0,
                  "unsupported_certification_max": 0, "regression_errors_max": 0, "sentinel_changed_max": 0},
        "runtime_labels_available": False, "pilot_allowed": False,
        "scope": "Bounded atomic seeker wants/likes states and utterances, coarse current/past time. "
                 "Runtime parser produces scope without gold metadata; unsupported compound/universal assertions "
                 "remain in stress evaluation as abstentions. Not arbitrary-language accuracy; not directly comparable "
                 "to prior 81-instance protocols. No recommendation-effect claim.",
        "stop": "one fixed batch, no automatic retry/expansion/prompt change"}
    plan["plan_hash"] = fingerprint(plan)
    create(output / "plan.json", plan)
    for p in files:
        dest = output / "source_snapshot" / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.read_bytes() != p.read_bytes():
            raise ValueError("snapshot changed")
        dest.write_bytes(p.read_bytes())
    return {"status": "prepared", "primary": 40, "stress": 8, "regressions": len(old), "maximum_requests": maximum_requests,
            "new_api_calls": 0, "plan_hash": plan["plan_hash"]}


def verify_plan(output):
    output = Path(output)
    plan = json.loads((output / "plan.json").read_text())
    if plan["plan_hash"] != fingerprint({k: v for k, v in plan.items() if k != "plan_hash"}):
        raise ValueError("plan changed")
    for name, expected in plan["code_sha256"].items():
        if sha(ROOT / name) != expected or sha(output / "source_snapshot" / name) != expected:
            raise ValueError("runtime source or snapshot changed")
    for name, key in (("inputs.json", "inputs_hash"), ("regression_inputs.json", "regression_hash")):
        if fingerprint(json.loads((output / name).read_text())) != plan[key]:
            raise ValueError("runtime input changed")
    return plan


def infer(row, client):
    state = StateV6(row["reports"], row["evidence"])
    target, sentinel = "current_request:c1", "current_request:sentinel"
    sources = source_map(row["evidence"])
    def patch(key):
        return {"target_claim_id": key, "expected_hash": fingerprint(state.claims[key]), "replacement": None}
    state.apply("current_request", sentinel, patch(sentinel), EvidenceVerifier(sources))
    before = copy.deepcopy(state.claims[sentinel])
    pipeline = AtomicEvidencePipeline(sources, client)
    try:
        check = pipeline.verify(state.claims[target]["claim"], row["id"])
    except (ValueError, KeyError, TypeError) as exc:
        check = {"verdict": "insufficient", "kind": "invalid_proxy", "verification_failed": True,
                 "reason": str(exc), "atom": pipeline.last_atom}
    class Checked:
        def verify(self, claim): return copy.deepcopy(check)
    event = state.apply("current_request", target, patch(target), Checked())
    return {"verification": check, "action": event["action"], "sentinel_preserved": before == state.claims[sentinel],
            "state": state.payload()}


def faithful(predicted, expected):
    if not isinstance(predicted, dict): return False
    projected = {k: predicted.get(k) for k in expected}
    if isinstance(projected.get("object"), str):
        projected["object"] = " ".join(projected["object"].casefold().split())
    expected = copy.deepcopy(expected)
    if isinstance(expected.get("object"), str):
        expected["object"] = " ".join(expected["object"].casefold().split())
    return projected == expected


def score(output, plan):
    output = Path(output)
    labels = json.loads((output / "labels.json").read_text())
    if fingerprint(labels) != plan["labels_hash"]:
        raise ValueError("labels changed")
    counts, classes, correct_classes = Counter(), Counter(), Counter()
    failures = []
    for row in labels:
        folder = "regression" if row["category"].endswith("_regression") else "new"
        pred = json.loads((output / "predictions" / folder / (row["id"] + ".json")).read_text())
        check = pred["verification"]
        atom_ok = faithful(check.get("atom"), row["expected_atom"])
        valid = not check.get("verification_failed", False)
        correct = check["verdict"] == row["label"]
        counts["invalid"] += not valid
        counts["sentinel_changed"] += not pred["sentinel_preserved"]
        counts["interpretation_errors"] += not atom_ok
        if folder == "regression":
            counts["regression_errors"] += not (correct and valid and atom_ok)
        elif row["supported_scope"]:
            counts["primary_items"] += 1
            counts["primary_correct"] += correct
            counts["primary_joint_correct"] += correct and atom_ok and valid
            counts["false_supported"] += check["verdict"] == "supported" and row["label"] != "supported"
            classes[row["label"]] += 1
            correct_classes[row["label"]] += correct
            if row["category"] == "conflict_preference_state":
                counts["unresolved_decisive"] += check["verdict"] != "insufficient"
        else:
            counts["unsupported_items"] += 1
            counts["unsupported_certification"] += check.get("certification_attempted", True) or not atom_ok or not valid
        if valid and row["supported_scope"]:
            relations = {r["source_id"]: r["relation"] for r in check.get("source_relations", [])}
            counts["source_relations"] += len(row["relations"])
            counts["correct_source_relations"] += sum(relations.get(sid) == value for sid, value in row["relations"].items())
        if not (correct and atom_ok and valid and pred["sentinel_preserved"]):
            failures.append({"id": row["id"], "category": row["category"], "label": row["label"],
                             "expected_atom": row["expected_atom"], "prediction": check})
    rates = {label: correct_classes[label] / n for label, n in classes.items()}
    accuracy = counts["primary_correct"] / counts["primary_items"]
    g = plan["gates"]
    gates = {"primary_accuracy": accuracy >= g["primary_accuracy_min"],
        "class_recall": min(rates.values()) >= g["recall_each_min"],
        **{key: counts[key] <= g[key + "_max"] for key in ("false_supported", "interpretation_errors", "invalid",
           "unresolved_decisive", "unsupported_certification", "regression_errors", "sentinel_changed")}}
    raw = [json.loads(p.read_text()) for p in (output / "slots").glob("*.raw.json")]
    bodies = [json.loads(r["raw"]["body"]) for r in raw]
    models = sorted({r.get("model") for r in bodies})
    pending = len(list((output / "slots").glob("*.intent.json")))
    gates["one_model_no_pending"] = len(models) == 1 and pending == 0
    result = {"pass": all(gates.values()), "gates": gates, "counts": dict(counts), "primary_accuracy": accuracy,
        "class_recall": rates, "primary_joint_accuracy": counts["primary_joint_correct"] / counts["primary_items"],
        "supported_domain_coverage": counts["primary_items"] / (counts["primary_items"] + counts["unsupported_items"]),
        "failures": failures, "plan_hash": plan["plan_hash"], "contract_version": "atomic_v3", "pilot_allowed": False,
        "physical": {"responses": len(raw), "pending_intents": pending, "models": models,
                     "prompt_tokens": sum(b["usage"]["prompt_tokens"] for b in bodies),
                     "completion_tokens": sum(b["usage"]["completion_tokens"] for b in bodies)}, "scope": plan["scope"]}
    create(output / "release_score.json", result)
    return result


def execute(output):
    output = Path(output)
    plan = verify_plan(output)
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    client = AttemptClient(DurableProvider(output / "slots", plan, RawDeepSeek(model=plan["model"]), parse_raw, True), plan["limits"])
    for name, folder in (("inputs.json", "new"), ("regression_inputs.json", "regression")):
        for row in json.loads((output / name).read_text()):
            create(output / "predictions" / folder / (row["id"] + ".json"), infer(row, client))
    return score(output, plan)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-authorized", action="store_true")
    args = parser.parse_args()
    print(json.dumps(execute(args.output) if args.execute_authorized else prepare(args.output), ensure_ascii=False, indent=2))
