"""Create a reproducible inventory of existing MAS experiment assets.

This command is read-only with respect to source data.  It writes audit reports
under results/audit and never copies API keys or raw prompt content.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def structural_summary(value: Any) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"root_type": type(value).__name__}
    if isinstance(value, dict):
        summary["root_keys"] = sorted(value.keys())
        for key in ("questions", "experiments", "papers", "online_experiments"):
            items = value.get(key)
            if isinstance(items, list):
                summary[f"n_{key}"] = len(items)
    elif isinstance(value, list):
        summary["n_items"] = len(value)
    return summary


def record_ids(items: Iterable[dict], id_fields: List[str]) -> List[str]:
    ids: List[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        parts = [str(item.get(key, "")) for key in id_fields]
        ids.append("|".join(parts))
    return ids


def check_duplicates(value: Any) -> Dict[str, int]:
    checks = [
        ("questions", ["question_id"]),
        ("papers", ["paper_id"]),
        ("experiments", ["dataset", "condition", "question_id"]),
        ("online_experiments", ["dataset", "condition", "question_id"]),
    ]
    out: Dict[str, int] = {}
    if not isinstance(value, dict):
        return out
    for key, fields in checks:
        items = value.get(key)
        if not isinstance(items, list):
            continue
        ids = record_ids(items, fields)
        out[f"duplicate_{key}"] = sum(n - 1 for n in Counter(ids).values() if n > 1)
        out[f"error_{key}"] = sum(1 for x in items if isinstance(x, dict) and "error" in x)
    return out


def inventory(results_dir: Path) -> Dict[str, Any]:
    files = []
    for path in sorted(results_dir.rglob("*")):
        if not path.is_file() or "audit" in path.relative_to(results_dir).parts:
            continue
        item: Dict[str, Any] = {
            "path": str(path.relative_to(ROOT)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        if path.suffix.lower() == ".json":
            value = load_json(path)
            item["json_valid"] = value is not None
            if value is not None:
                item.update(structural_summary(value))
                item.update(check_duplicates(value))
        files.append(item)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT),
        "n_files": len(files),
        "total_bytes": sum(x["size_bytes"] for x in files),
        "files": files,
    }


def consistency_report(inv: Dict[str, Any]) -> Dict[str, Any]:
    invalid = [x["path"] for x in inv["files"] if x.get("json_valid") is False]
    duplicates = {
        x["path"]: {k: v for k, v in x.items() if k.startswith("duplicate_") and v}
        for x in inv["files"]
        if any(k.startswith("duplicate_") and v for k, v in x.items())
    }
    errors = {
        x["path"]: {k: v for k, v in x.items() if k.startswith("error_") and v}
        for x in inv["files"]
        if any(k.startswith("error_") and v for k, v in x.items())
    }
    sample_sizes = {}
    for item in inv["files"]:
        counts = {k: v for k, v in item.items() if k.startswith("n_")}
        if counts:
            sample_sizes[item["path"]] = counts
    return {
        "invalid_json": invalid,
        "duplicate_records": duplicates,
        "error_records": errors,
        "sample_sizes": sample_sizes,
        "structural_integrity_pass": not invalid,
        "legacy_paired_reanalysis_ready": not invalid and not duplicates and not errors,
        "pass": not invalid,
        "notes": [
            "A passing structural audit does not establish train/test independence.",
            "Old summary-only files cannot support paired causal reanalysis.",
            "API errors remain incomplete records and must not enter denominators.",
            "Disclosed duplicates in archived/legacy files are warnings, not a reason to mutate history.",
        ],
    }


def lineage_report(inv: Dict[str, Any], consistency: Dict[str, Any]) -> str:
    duplicate_paths = sorted(consistency["duplicate_records"])
    error_paths = sorted(consistency["error_records"])
    return f"""# AEIC-MD data lineage

Generated by `experiments/audit_experiment_assets.py`. Source assets are never rewritten.

## Lineage classes

| Asset class | Source | Downstream role | Admissibility |
|---|---|---|---|
| Frozen QA questions | `results/cache/StrategyQA.json`, `MMLU-Hard.json`, `TruthfulQA.json` + `experiments/mechanism/splits.json` | deterministic train/dev/test sampling | admissible; test IDs are fixed before collection |
| New follower-response records | `results/mechanism/pilot/`, `formal/` | E1–E6 question/agent/round analysis | admissible only when `completed=true`, verifier passes, and unique key is intact |
| Legacy topology/signals | `results/cache/*_*.json`, `results/signals/` | historical information-structure baselines | descriptive reuse; API-error entries excluded from denominators |
| Legacy online/optimization | `results/online_stackelberg/`, `results/archive/` | negative results and method history | not evidence of causal follower response; duplicates/errors retained and disclosed |
| Review-domain studies | `results/review/` | natural pathology and review-domain boundary evidence | agreement/behavior endpoints only; no truthfulness claim from official decisions |
| Frozen bundles | `results/mechanism/frozen/` | paper tables and one-command recomputation | admissible only after SHA256 and matrix verification pass |

## Audit disposition

- Inventoried files: {inv['n_files']}.
- Invalid JSON files: {len(consistency['invalid_json'])}.
- Legacy files containing duplicate keys: {len(duplicate_paths)}.
- Files containing explicit error records: {len(error_paths)}.
- Structural integrity pass: `{str(consistency['structural_integrity_pass']).lower()}`.
- Legacy paired reanalysis ready: `{str(consistency['legacy_paired_reanalysis_ready']).lower()}`.

Duplicate and error-bearing legacy assets are not repaired in place. They remain traceable in
`result_consistency.json`; new AEIC-MD confirmatory claims use only verifier-approved question-level
records. Summary-only legacy files cannot enter paired causal tests.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--out", type=Path, default=RESULTS / "audit")
    args = parser.parse_args()

    inv = inventory(args.results)
    consistency = consistency_report(inv)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "asset_inventory.json").write_text(
        json.dumps(inv, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out / "result_consistency.json").write_text(
        json.dumps(consistency, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out / "data_lineage.md").write_text(
        lineage_report(inv, consistency), encoding="utf-8"
    )
    print(f"Audited {inv['n_files']} files ({inv['total_bytes']} bytes)")
    print(f"Invalid JSON: {len(consistency['invalid_json'])}")
    print(f"Files with duplicate record keys: {len(consistency['duplicate_records'])}")
    print(f"Files with recorded errors: {len(consistency['error_records'])}")
    print(f"Structural audit pass: {consistency['structural_integrity_pass']}")
    print(f"Legacy paired reanalysis ready: {consistency['legacy_paired_reanalysis_ready']}")


if __name__ == "__main__":
    main()
