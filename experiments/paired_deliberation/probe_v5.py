"""Frozen 24-item verifier calibration and three-case V5 engineering probe."""
import argparse
import copy
import json
from collections import Counter
from pathlib import Path

from experiments.consensus_recommendation.common import ROOT, OUTPUT, create, read_rows, sha, fingerprint
from experiments.consensus_recommendation.protocol import bundle
from experiments.paired_deliberation.client_v5 import BoundedClient, RawDeepSeek, parse_raw
from experiments.paired_deliberation.durable_provider import DurableProvider
from experiments.paired_deliberation.pipeline_v5 import run_case
from experiments.paired_deliberation.requests_v5 import RequestsV5
from experiments.paired_deliberation.run_experiment import load_case_ids
from experiments.paired_deliberation.verified_state import EvidenceVerifier


def fixtures(packet):
    """Bounded evidence-relative NL calibration; labels never enter judge payload."""
    movies = {}
    for case in packet["cases"]:
        for movie in case["candidate_facts"]:
            if type(movie.get("year")) is int and movie["movie_id"] not in movies:
                movies[movie["movie_id"]] = movie
    rows = []
    for index, movie in enumerate(list(movies.values())[:8]):
        mid, year = movie["movie_id"], movie["year"]
        claims = [
            (f"According to the supplied catalogue, film {mid} was released in {year}.", "supported"),
            (f"According to the supplied catalogue, film {mid} was released in {year + 1}.", "refuted"),
            (f"The person requesting recommendations has watched and enjoyed film {mid}.", "insufficient"),
        ]
        for kind, (text, label) in enumerate(claims):
            rows.append({"id": "cal_%02d_%d" % (index, kind),
                         "input": {"claim": {"claim_id": "c1", "text": text,
                                   "source_ids": [movie["source_id"]], "candidate_ids": [mid], "stance": "support"},
                                   "evidence": {movie["source_id"]: copy.deepcopy(movie)}},
                         "label": label})
    if len(rows) != 24:
        raise ValueError("calibration fixture pool is too small")
    return rows


def run(output, execute=False, reuse_calibration=None, reuse_initial=None, reuse_exact=None):
    output = Path(output)
    packet_path = ROOT / "experiments/results/paired_deliberation_preflight/blind_annotation_packet_dev12.json"
    rows = fixtures(json.loads(packet_path.read_text()))
    ids = load_case_ids("engineering")[:3]
    limits = {"per_call_prompt_tokens": 64000, "per_call_completion_tokens": 1280,
              "total_prompt_tokens": 260000, "total_completion_tokens": 40000}
    source_names = ["probe_v5.py", "pipeline_v5.py", "verified_state.py", "requests_v5.py",
                    "durable_provider.py", "client_v5.py"]
    plan = {"version": "v5-probe-24nl-3engineering-v1", "case_ids": ids,
            "calibration_items": 24, "calibration_inputs_sha256": fingerprint([x["input"] for x in rows]),
            "model": "deepseek-chat", "limits": limits,
            "gates": {"accuracy_min": .90, "recall_each_class_min": .75, "false_supported_max": 0,
                      "engineering_cases_all_valid": 3, "invalid_semantic_verifications_max": 0},
            "natural_dialogue_accuracy_measured": False, "automatic_pilot_allowed": False,
            "source_sha256": {name: sha(Path(__file__).parent / name) for name in source_names},
            "input_sha256": {name: sha(OUTPUT / "data" / name)
                             for name in ("dev.inputs.jsonl", "dev.candidates.jsonl")},
            "known_label_scope": "controlled NL paraphrases and insufficient-evidence cases only"}
    if reuse_calibration:
        reuse_calibration = Path(reuse_calibration)
        plan["calibration_replay_source"] = str(reuse_calibration)
        prior_plan = json.loads((reuse_calibration / "plan.json").read_text())
        if prior_plan["calibration_inputs_sha256"] != plan["calibration_inputs_sha256"]:
            raise ValueError("calibration inputs changed; cannot replay")
        plan["calibration_replay_plan_sha256"] = sha(reuse_calibration / "plan.json")
    if reuse_initial:
        plan["initial_replay_root"] = str(reuse_initial)
    if reuse_exact:
        plan["exact_request_replay_root"] = str(reuse_exact)
    create(output / "plan.json", plan)
    create(output / "calibration.inputs.json", [{"id": row["id"], **row["input"]} for row in rows])
    create(output / "calibration.labels.json", [{"id": row["id"], "label": row["label"]} for row in rows])
    if not execute:
        return {"status": "prepared", "calibration_items": 24, "engineering_cases": 3, "new_api_calls": 0}
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    provider = DurableProvider(output / "slots", plan, RawDeepSeek(model=plan["model"]), parse_raw, True,
                               Path(reuse_exact or reuse_initial) / "slots" if (reuse_exact or reuse_initial) else None,
                               reuse_all=bool(reuse_exact))
    calibration_client = BoundedClient(provider, {**limits, "total_prompt_tokens": 260000,
                                                "total_completion_tokens": 40000})
    scored = []
    for row in rows:
        record_path = output / "calibration" / (row["id"] + ".json")
        def judge(payload, slot=row["id"]):
            request = RequestsV5.verification(payload)
            if reuse_calibration:
                saved = json.loads((reuse_calibration / "slots" / (slot + ".raw.json")).read_text())
                if saved["request"] != request or saved["integrity"] != fingerprint(saved["raw"]):
                    raise ValueError("raw replay request changed or response tampered")
                response = parse_raw(saved["raw"], request)
            else:
                response = calibration_client.complete(slot, request)
            return {**response["report"], "usage": response["usage"]}
        try:
            result = EvidenceVerifier(row["input"]["evidence"], judge).verify(row["input"]["claim"])
            item = {"id": row["id"], "result": result, "label": row["label"],
                    "correct": result["verdict"] == row["label"], "status": "valid"}
        except (ValueError, KeyError, TypeError) as exc:
            item = {"id": row["id"], "label": row["label"], "correct": False,
                    "status": "invalid", "error": str(exc)}
        create(record_path, item)
        scored.append(item)
        print(json.dumps({"calibration_completed": len(scored)}), flush=True)
    # Gate is scored only after inference. No evaluation label is sent to a judge.
    recalls = {label: sum(x["correct"] for x in scored if x["label"] == label) / 8
               for label in ("supported", "refuted", "insufficient")}
    accuracy = sum(x["correct"] for x in scored) / 24
    false_supported = sum(x.get("result", {}).get("verdict") == "supported" and x["label"] != "supported" for x in scored)
    calibration_pass = accuracy >= .9 and min(recalls.values()) >= .75 and false_supported == 0
    inputs = {x["case_id"]: x for x in read_rows(OUTPUT / "data/dev.inputs.jsonl")}
    candidates = {x["case_id"]: x for x in read_rows(OUTPUT / "data/dev.candidates.jsonl")}
    _, _, movies = __import__("experiments.scripts.data.build_redial_benchmark", fromlist=["load_catalog"]).load_catalog(
        ROOT / "experiments/data/processed/movie_catalog.jsonl")
    engineering = []
    if calibration_pass:
        for cid in ids:
            path = output / "cases" / (cid.replace(":", "_") + ".json")
            if path.exists():
                entry = json.loads(path.read_text())
            else:
                try:
                    result = run_case(bundle(inputs[cid], candidates[cid], movies),
                                      BoundedClient(provider, limits), use_model_verifier=True)
                    entry = {"case_id": cid, "status": "valid", "result": result}
                except (ValueError, KeyError, TypeError) as exc:
                    entry = {"case_id": cid, "status": "invalid", "error": str(exc)}
                create(path, entry)
            engineering.append(entry)
            print(json.dumps({"engineering_completed": len(engineering), "status": entry["status"]}), flush=True)
    counts, tokens, models = Counter(), Counter(), Counter()
    for path in (output / "slots").glob("*.json"):
        if path.name.endswith((".raw.json", ".intent.json")):
            continue
        record = json.loads(path.read_text())["record"]
        counts["parsed_responses"] += 1
        models[record["model"]] += 1
        for name in ("prompt_tokens", "completion_tokens"):
            tokens[name] += record["usage"][name]
    replayed = [json.loads(path.read_text()) for path in (output / "slots").glob("*.raw.json")
                if "reused_from" in json.loads(path.read_text())]
    reused_initial_count = sum(x["request"]["kind"] == "initial" for x in replayed)
    verifier_failures = sum(bool(event["old_verification"].get("verification_failed"))
                            for entry in engineering if entry["status"] == "valid"
                            for event in entry["result"]["events"])
    summary = {"calibration_accuracy": accuracy, "recall_by_class": recalls,
               "false_supported": false_supported, "calibration_pass": calibration_pass,
               "engineering_valid": sum(x["status"] == "valid" for x in engineering),
               "engineering_attempted": len(engineering),
               "engineering_pass": len(engineering) == 3 and all(x["status"] == "valid" for x in engineering),
               "usage": {**dict(counts), **dict(tokens), "models": dict(models)},
               "natural_dialogue_accuracy_measured": False, "recommendation_gain_measured": False,
               "automatic_pilot_allowed": False, "reserved_labels_read": False,
               "invalid_semantic_verifications": verifier_failures,
               "semantic_engineering_gate": len(engineering) == 3 and verifier_failures == 0
                                             and all(x["status"] == "valid" for x in engineering),
               "plan_sha256": sha(output / "plan.json")}
    summary["reused_calibration_responses"] = 24 if reuse_calibration else 0
    summary["reused_initial_responses"] = reused_initial_count
    summary["reused_exact_engineering_responses"] = len(replayed)
    create(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-authorized", action="store_true")
    parser.add_argument("--reuse-calibration", type=Path)
    parser.add_argument("--reuse-initial", type=Path)
    parser.add_argument("--reuse-exact", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.execute_authorized, args.reuse_calibration, args.reuse_initial,
                         args.reuse_exact), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
