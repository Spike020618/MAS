"""Build and verify the terminal manifest across immutable AEIC-MD bundles."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from .recompute_frozen import comparable
except ImportError:  # Direct script execution.
    from recompute_frozen import comparable


REQUIRED = (
    "matrix.json", "question_level.jsonl", "analysis.json", "statistics.json",
    "sensitivity.json", "verification.json", "provenance.json",
)
RECOMPUTED = ("analysis.json", "statistics.json", "sensitivity.json", "comparison.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def inspect_bundle(bundle: Path) -> dict:
    missing = [name for name in REQUIRED if not (bundle / name).is_file()]
    verification = json_load(bundle / "verification.json") if not missing else {}
    provenance = json_load(bundle / "provenance.json") if not missing else {}
    recomputed = bundle.with_name(bundle.name + ".recomputed")
    recomputed_required = list(RECOMPUTED)
    if (bundle / "group_analysis.json").is_file():
        recomputed_required.append("group_analysis.json")
    if (bundle / "group_statistics.json").is_file():
        recomputed_required.append("group_statistics.json")
    if (bundle / "planned_contrasts.json").is_file():
        recomputed_required.append("planned_contrasts.json")
    if (bundle / "behavioral_manipulation_tests.json").is_file():
        recomputed_required.append("behavioral_manipulation_tests.json")
    recompute_missing = [name for name in recomputed_required
                         if not (recomputed / name).is_file()]
    mismatches = []
    if not missing and not recompute_missing:
        result_names = ["analysis.json", "statistics.json", "sensitivity.json"]
        if (bundle / "group_analysis.json").is_file():
            result_names.append("group_analysis.json")
        if (bundle / "group_statistics.json").is_file():
            result_names.append("group_statistics.json")
        if (bundle / "planned_contrasts.json").is_file():
            result_names.append("planned_contrasts.json")
        if (bundle / "behavioral_manipulation_tests.json").is_file():
            result_names.append("behavioral_manipulation_tests.json")
        for name in result_names:
            if comparable(json_load(bundle / name)) != comparable(json_load(recomputed / name)):
                mismatches.append(name)
        comparison = json_load(recomputed / "comparison.json")
        if not comparison or not all(comparison.values()):
            mismatches.append("comparison.json")
    return {
        "name": bundle.name,
        "path": str(bundle),
        "matrix_version": verification.get("matrix_version"),
        "n_question_level_records": provenance.get("n_question_level_records"),
        "verification_pass": verification.get("pass") is True,
        "missing_files": missing,
        "recomputed_path": str(recomputed),
        "recompute_missing_files": recompute_missing,
        "recompute_mismatches": mismatches,
        "question_level_sha256": (
            sha256(bundle / "question_level.jsonl") if not missing else None
        ),
        "complete": (
            not missing and verification.get("pass") is True
            and not recompute_missing and not mismatches
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("experiments/results/mechanism/frozen"))
    parser.add_argument("--require", action="append", default=[],
                        help="Required bundle name; repeat for every preregistered suite")
    parser.add_argument("--out", type=Path,
                        default=Path("experiments/results/mechanism/frozen/manifest.json"))
    args = parser.parse_args()
    bundles = sorted(
        path for path in args.root.iterdir()
        if path.is_dir() and not path.name.endswith(".recomputed")
    ) if args.root.exists() else []
    items = [inspect_bundle(path) for path in bundles]
    present = {item["name"] for item in items}
    missing_suites = sorted(set(args.require) - present)
    report = {
        "required_suites": args.require,
        "missing_suites": missing_suites,
        "bundles": items,
        "total_question_level_records": sum(
            int(item["n_question_level_records"] or 0) for item in items
        ),
    }
    report["pass"] = not missing_suites and all(item["complete"] for item in items)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
