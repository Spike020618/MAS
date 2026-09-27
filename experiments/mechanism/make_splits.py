"""Create deterministic, ground-truth-stratified train/dev/test IDs."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATASETS = ["StrategyQA", "MMLU-Hard", "TruthfulQA"]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_dataset(path: Path, seed: int) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    strata = {}
    for index, item in enumerate(payload.get("questions", [])):
        if item.get("error") or not item.get("question") or not item.get("ground_truth"):
            continue
        gt = str(item["ground_truth"]).lower()
        strata.setdefault(gt, []).append(str(item.get("question_id", index)))
    rng = random.Random(seed)
    for ids in strata.values():
        rng.shuffle(ids)

    targets = {"train": 100, "dev": 50, "test": 150}
    splits = {name: [] for name in targets}
    # Exact proportional allocation within every split.
    total = sum(len(ids) for ids in strata.values())
    if sum(targets.values()) > total:
        raise ValueError(f"{path} has {total} valid questions; need {sum(targets.values())}")
    offsets = {gt: 0 for gt in strata}
    labels = sorted(strata)
    target_items = list(targets.items())
    for split_index, (name, size) in enumerate(target_items):
        if split_index == len(target_items) - 1:
            allocation = {gt: len(strata[gt]) - offsets[gt] for gt in labels}
        else:
            raw = {gt: size * len(strata[gt]) / total for gt in labels}
            allocation = {gt: int(raw[gt]) for gt in labels}
            remainder = size - sum(allocation.values())
            for gt in sorted(labels, key=lambda x: raw[x] - allocation[x], reverse=True)[:remainder]:
                allocation[gt] += 1
        chosen = []
        for gt in labels:
            start = offsets[gt]
            end = start + allocation[gt]
            chosen.extend(strata[gt][start:end])
            offsets[gt] = end
        rng.shuffle(chosen)
        splits[name] = chosen
    return {
        "source": str(path.relative_to(ROOT)),
        "source_sha256": file_sha256(path),
        "n_valid": total,
        "splits": splits,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).with_name("splits.json"))
    args = parser.parse_args()
    output = {"version": "1.0", "seed": args.seed, "datasets": {}}
    for dataset in DATASETS:
        path = ROOT / "results" / "cache" / f"{dataset}.json"
        output["datasets"][dataset] = split_dataset(path, args.seed)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
