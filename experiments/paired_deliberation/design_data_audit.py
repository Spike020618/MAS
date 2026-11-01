"""D1 input-only exposure audit; never opens label or reserved-test files."""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from experiments.consensus_recommendation.common import OUTPUT, ROOT, create, fingerprint, read_rows, sha


def text_key(row):
    return fingerprint([(obs["role"], re.sub(r"\s+", " ", obs["text"].strip().casefold()))
                        for obs in row["observations"]])


def audit(output):
    data = OUTPUT / "data"
    train = read_rows(data / "train.inputs.jsonl")
    dev = read_rows(data / "dev.inputs.jsonl")
    candidates = {row["case_id"]: row for row in read_rows(data / "dev.candidates.jsonl")}
    retrieval = json.loads((data / "development_retrieval_audit.json").read_text())
    train_keys, dev_keys = defaultdict(list), defaultdict(list)
    for row in train:
        train_keys[text_key(row)].append(row["case_id"])
    for row in dev:
        dev_keys[text_key(row)].append(row["case_id"])
    known = set(retrieval["engineering_case_ids"]) | set(retrieval["pilot_case_ids"])
    shape_errors = []
    for row in dev:
        cid = row["case_id"]
        if [obs["message_index"] for obs in row["observations"]] != list(range(row["cutoff_exclusive"])):
            shape_errors.append({"case": cid, "error": "prefix is not a contiguous pre-cutoff sequence"})
        ranked = candidates[cid]["ranked_movie_ids"]
        if len(set(ranked)) != len(ranked) or len(ranked) != 100:
            shape_errors.append({"case": cid, "error": "candidate pool is not 100 unique IDs"})
    cross = [{"train": train_keys[key], "dev": dev_keys[key]} for key in sorted(set(train_keys) & set(dev_keys))]
    unknown = sorted({row["case_id"] for row in dev} - known)
    result = {"train_inputs": len(train), "dev_inputs": len(dev), "candidate_errors": shape_errors,
        "cross_partition_normalized_duplicate_groups": cross,
        "dev_internal_duplicate_groups": [ids for ids in dev_keys.values() if len(ids) > 1],
        "known_engineering_or_pilot_cases": len(known),
        "not_in_known_case_lists": unknown,
        "unexposed_case_count": None,
        "unexposed_note": "Absence from these two case lists is not proof of no exposure across other historical tasks. "
                          "Do not automatically promote remaining development cases to confirmatory test.",
        "reserved_files_opened": False, "label_files_opened": False,
        "source_sha256": {str((data / name).relative_to(ROOT)): sha(data / name) for name in (
            "train.inputs.jsonl", "dev.inputs.jsonl", "dev.candidates.jsonl", "development_retrieval_audit.json")},
        "future_task_metric": "dataset target retrieval, not satisfaction or latent preference truth",
        "next_formal_gate": "freeze independent sample provenance, label definition, grouping, sample size and analysis before scoring",
        "new_api_calls": 0}
    create(Path(output) / "input_exposure_audit.json", result)
    return {key: value for key, value in result.items() if key not in {"not_in_known_case_lists", "source_sha256"}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(audit(parser.parse_args().output), ensure_ascii=False, indent=2))
