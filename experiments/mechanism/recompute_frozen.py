"""Verify hashes and rebuild all derived numbers in a frozen AEIC-MD bundle."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def comparable(payload: dict) -> dict:
    """Remove invocation-location metadata while preserving every result field."""
    normalized = copy.deepcopy(payload)
    for key in ("records", "records_path"):
        normalized.pop(key, None)
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--out", type=Path,
                        help="Derived-output directory; defaults beside the bundle")
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    provenance = json.loads((bundle / "provenance.json").read_text(encoding="utf-8"))
    bad = []
    for relative, expected in provenance["files"].items():
        path = bundle / relative
        actual = sha256(path) if path.exists() else None
        if actual != expected:
            bad.append({"file": relative, "expected": expected, "actual": actual})
    if bad:
        raise SystemExit("frozen hash verification failed: " + json.dumps(bad))
    if args.check_only:
        print(f"verified {len(provenance['files'])} files")
        return
    canonical = bundle / "question_level.jsonl"
    target = (args.out or bundle.with_name(bundle.name + ".recomputed")).resolve()
    if target.exists():
        raise SystemExit(f"refusing to overwrite recomputed output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{target.name}.", dir=target.parent) as tmp:
        stage = Path(tmp)
        commands = [
            [sys.executable, str(HERE / "analyze.py"), str(canonical),
             "--out", str(stage / "analysis.json")],
            [sys.executable, str(HERE / "statistical_tests.py"), str(canonical),
             "--reference", provenance["reference"],
             "--out", str(stage / "statistics.json")],
            [sys.executable, str(HERE / "sensitivity.py"), str(canonical),
             "--out", str(stage / "sensitivity.json")],
        ]
        if (bundle / "group_analysis.json").exists():
            commands.append([
                sys.executable, str(HERE / "group_analysis.py"), str(canonical),
                "--reference", provenance["reference"],
                "--out", str(stage / "group_analysis.json"),
            ])
        if (bundle / "group_statistics.json").exists():
            commands.append([
                sys.executable, str(HERE / "group_statistical_tests.py"), str(canonical),
                "--reference", provenance["reference"],
                "--out", str(stage / "group_statistics.json"),
            ])
        if (bundle / "planned_contrasts.json").exists():
            commands.append([
                sys.executable, str(HERE / "planned_contrasts.py"),
                str(bundle / "matrix.json"), str(canonical),
                "--out", str(stage / "planned_contrasts.json"),
            ])
        if (bundle / "behavioral_manipulation_tests.json").exists():
            commands.append([
                sys.executable, str(HERE / "behavioral_manipulation_tests.py"),
                str(bundle / "matrix.json"), str(canonical),
                "--out", str(stage / "behavioral_manipulation_tests.json"),
            ])
        for command in commands:
            completed = subprocess.run(command, cwd=ROOT)
            if completed.returncode:
                raise SystemExit(completed.returncode)
        comparison = {
            "analysis_matches_frozen": comparable(json.loads((stage / "analysis.json").read_text()))
            == comparable(json.loads((bundle / "analysis.json").read_text())),
            "statistics_matches_frozen": comparable(json.loads((stage / "statistics.json").read_text()))
            == comparable(json.loads((bundle / "statistics.json").read_text())),
            "sensitivity_matches_frozen": comparable(json.loads((stage / "sensitivity.json").read_text()))
            == comparable(json.loads((bundle / "sensitivity.json").read_text())),
        }
        if (bundle / "group_analysis.json").exists():
            comparison["group_analysis_matches_frozen"] = (
                comparable(json.loads((stage / "group_analysis.json").read_text()))
                == comparable(json.loads((bundle / "group_analysis.json").read_text()))
            )
        if (bundle / "group_statistics.json").exists():
            comparison["group_statistics_matches_frozen"] = (
                comparable(json.loads((stage / "group_statistics.json").read_text()))
                == comparable(json.loads((bundle / "group_statistics.json").read_text()))
            )
        if (bundle / "planned_contrasts.json").exists():
            comparison["planned_contrasts_match_frozen"] = (
                comparable(json.loads((stage / "planned_contrasts.json").read_text()))
                == comparable(json.loads((bundle / "planned_contrasts.json").read_text()))
            )
        if (bundle / "behavioral_manipulation_tests.json").exists():
            comparison["behavioral_manipulation_tests_match_frozen"] = (
                comparable(json.loads((stage / "behavioral_manipulation_tests.json").read_text()))
                == comparable(json.loads((bundle / "behavioral_manipulation_tests.json").read_text()))
            )
        (stage / "comparison.json").write_text(
            json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
        )
        if not all(comparison.values()):
            raise SystemExit("recomputed outputs do not match frozen outputs")
        os.replace(stage, target)
    print(target)


if __name__ == "__main__":
    main()
