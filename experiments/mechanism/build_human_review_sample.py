"""Build a deterministic, protocol-blind sample for report-interface review."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--per-protocol", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    questions = {}
    for path in args.cache_dir.glob("*.json"):
        dataset = path.stem
        payload = json.loads(path.read_text(encoding="utf-8"))
        for i, item in enumerate(payload.get("questions") or []):
            questions[(dataset, str(item.get("question_id", i)))] = item.get("question", "")
    groups = defaultdict(list)
    for line in args.records.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        row = json.loads(line)
        if row.get("completed") and not row.get("error"):
            groups[row["protocol"]].append(row)
    rng = random.Random(args.seed)
    selected = []
    for protocol, rows in sorted(groups.items()):
        by_domain = defaultdict(list)
        for row in rows: by_domain[row["dataset"]].append(row)
        quota = {name: args.per_protocol // len(by_domain) for name in by_domain}
        for name in sorted(by_domain)[:args.per_protocol % len(by_domain)]: quota[name] += 1
        for domain in sorted(by_domain):
            pool = sorted(by_domain[domain], key=lambda x: str(x["question_id"]))
            selected.extend(rng.sample(pool, quota[domain]))
    rng.shuffle(selected)
    items = []
    key = []
    for index, row in enumerate(selected, 1):
        blind_id = f"R{index:03d}"
        response = row["raw_response"]
        digest = hashlib.sha256(response.encode("utf-8")).hexdigest()[:12]
        items.append({"blind_id": blind_id, "dataset": row["dataset"],
                      "question_id": str(row["question_id"]),
                      "question": questions.get((row["dataset"], str(row["question_id"])), ""),
                      "ground_truth": row["ground_truth"], "response": response,
                      "response_hash": digest})
        key.append({"blind_id": blind_id, "protocol": row["protocol"],
                    "dataset": row["dataset"], "question_id": str(row["question_id"]),
                    "response_hash": digest})
    payload = {"seed": args.seed, "per_protocol": args.per_protocol,
               "n_items": len(items), "items": items, "blind_key": key}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"n_items": len(items), "out": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
