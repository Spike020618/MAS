"""Prepare V5 repair engineering; network requires explicit execution flag.

This entry validates the repaired mechanism. It does not claim to implement the
full E1-E5 research program or to establish proxy-verifier accuracy.
"""
import json
from pathlib import Path

from experiments.consensus_recommendation.common import ROOT, OUTPUT, create, fingerprint, read_rows, sha
from experiments.consensus_recommendation.protocol import bundle
from experiments.paired_deliberation.client_v5 import BoundedClient, RawDeepSeek, parse_raw
from experiments.paired_deliberation.durable_provider import DurableProvider
from experiments.paired_deliberation.pipeline_v5 import run_case
from experiments.paired_deliberation.run_experiment import load_case_ids


def run(stage, output, execute_authorized=False):
    if stage != "engineering":
        raise ValueError("V5 repair must pass engineering and semantic calibration before a new pilot protocol")
    output = Path(output)
    if (output / "plan.json").exists():
        old = json.loads((output / "plan.json").read_text())
        if old.get("version") != "claim-updates-v5":
            raise ValueError("historical experiment directory is immutable; choose a V5 output directory")
    limits = {"per_call_prompt_tokens": 64000, "per_call_completion_tokens": 1280,
              "total_prompt_tokens": 260000, "total_completion_tokens": 40000}
    source_files = ["verified_state.py", "requests_v5.py", "pipeline_v5.py", "client_v5.py",
                    "durable_provider.py", "run_v5.py"]
    plan = {"version": "claim-updates-v5", "stage": stage, "case_ids": load_case_ids(stage),
            "model": "deepseek-chat", "limits": limits, "semantic_verifier": "model_proxy_plus_exact_catalog",
            "input_sha256": {name: sha(OUTPUT / "data" / name)
                             for name in ("dev.inputs.jsonl", "dev.candidates.jsonl")},
            "code_sha256": {name: sha(Path(__file__).parent / name) for name in source_files},
            "shared_dependency_sha256": {
                "legacy_protocol": sha(Path(__file__).parents[1] / "consensus_recommendation/protocol.py"),
                "artifact_helpers": sha(Path(__file__).parents[1] / "consensus_recommendation/common.py"),
                "catalog_loader": sha(Path(__file__).parents[1] / "scripts/data/build_redial_benchmark.py")},
            "evaluation_labels_available": False,
            "scope": "V5 repair engineering only; full comparative research experiments not certified"}
    create(output / "plan.json", plan)
    if not execute_authorized:
        return {"status": "prepared", "cases": len(plan["case_ids"]), "new_api_calls": 0,
                "natural_language_proxy_accuracy_evaluated": False}
    acceptance = ROOT / "experiments/results/paired_deliberation_v5_repair/controlled_fact_acceptance.json"
    check = json.loads(acceptance.read_text())
    if not check["pass"] or check["cases"] != 288:
        raise ValueError("controlled repair acceptance missing")
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    transport = RawDeepSeek(model=plan["model"])
    provider = DurableProvider(output / "slots", plan, transport, parse_raw, True)
    inputs = {row["case_id"]: row for row in read_rows(OUTPUT / "data/dev.inputs.jsonl")}
    candidates = {row["case_id"]: row for row in read_rows(OUTPUT / "data/dev.candidates.jsonl")}
    _, _, movies = __import__("experiments.scripts.data.build_redial_benchmark", fromlist=["load_catalog"]).load_catalog(
        ROOT / "experiments/data/processed/movie_catalog.jsonl")
    for cid in plan["case_ids"]:
        case_path = output / "cases" / (cid.replace(":", "_") + ".json")
        if case_path.exists():
            saved = json.loads(case_path.read_text())
            if saved["plan_hash"] != fingerprint(plan) or saved["integrity"] != fingerprint(saved["record"]):
                raise ValueError("case checkpoint is stale or tampered")
            continue
        evidence = bundle(inputs[cid], candidates[cid], movies)
        result = run_case(evidence, BoundedClient(provider, limits), use_model_verifier=True)
        create(case_path, {"plan_hash": fingerprint(plan), "record": result, "integrity": fingerprint(result)})
    return {"status": "engineering_collected", "cases": len(plan["case_ids"]),
            "next_gate": "semantic proxy calibration and full baseline protocol required before pilot"}
