"""Freeze a completed matrix into an immutable, one-command reproducible bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.mechanism.schema import record_from_dict


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_unique_records(paths: list[Path]) -> list[dict]:
    records: dict[tuple, dict] = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                item = json.loads(line)
                key = (
                    item.get("experiment_id"), item.get("dataset"),
                    item.get("condition"), str(item.get("question_id")),
                    int(item.get("seed", -1)),
                )
                if key in records:
                    raise ValueError(f"duplicate unit {key} in {path}:{line_number}")
                if item.get("completed") is not True or item.get("error"):
                    raise ValueError(f"incomplete unit {key} in {path}:{line_number}")
                errors = record_from_dict(item).validate(require_complete=True)
                if errors:
                    raise ValueError(
                        f"invalid unit {key} in {path}:{line_number}: " + "; ".join(errors)
                    )
                records[key] = item
    return [records[key] for key in sorted(records)]


def run_checked(cmd: list[str]) -> None:
    completed = subprocess.run(cmd, cwd=ROOT)
    if completed.returncode:
        raise SystemExit(f"command failed ({completed.returncode}): {' '.join(cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    parser.add_argument("records", nargs="+", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out-root", type=Path,
                        default=ROOT / "results/mechanism/frozen")
    args = parser.parse_args()

    matrix = args.matrix.resolve()
    record_paths = [path.resolve() for path in args.records]
    target = (args.out_root / args.name).resolve()
    if target.exists():
        raise SystemExit(f"refusing to overwrite frozen bundle: {target}")
    args.out_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f".{args.name}.", dir=args.out_root) as tmp:
        stage = Path(tmp)
        verification = stage / "verification.json"
        run_checked([
            sys.executable, str(HERE / "verify_matrix.py"), str(matrix),
            *map(str, record_paths), "--out", str(verification),
        ])
        audit = json.loads(verification.read_text(encoding="utf-8"))
        if audit.get("pass") is not True:
            raise SystemExit("matrix verification failed; bundle was not frozen")

        raw_dir = stage / "raw"
        raw_dir.mkdir()
        shutil.copy2(matrix, stage / "matrix.json")
        for index, source in enumerate(record_paths):
            shutil.copy2(source, raw_dir / f"{index:02d}_{source.name}")

        canonical = stage / "question_level.jsonl"
        records = load_unique_records(record_paths)
        with canonical.open("w", encoding="utf-8") as handle:
            for item in records:
                handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")

        analysis = stage / "analysis.json"
        group_analysis = stage / "group_analysis.json"
        group_statistics = stage / "group_statistics.json"
        statistics = stage / "statistics.json"
        sensitivity = stage / "sensitivity.json"
        planned_contrasts = stage / "planned_contrasts.json"
        behavioral_tests = stage / "behavioral_manipulation_tests.json"
        run_checked([sys.executable, str(HERE / "analyze.py"), str(canonical),
                     "--out", str(analysis)])
        run_checked([sys.executable, str(HERE / "group_analysis.py"), str(canonical),
                     "--reference", args.reference, "--out", str(group_analysis)])
        run_checked([sys.executable, str(HERE / "group_statistical_tests.py"),
                     str(canonical), "--reference", args.reference,
                     "--out", str(group_statistics)])
        run_checked([sys.executable, str(HERE / "statistical_tests.py"), str(canonical),
                     "--reference", args.reference, "--out", str(statistics)])
        run_checked([sys.executable, str(HERE / "sensitivity.py"), str(canonical),
                     "--out", str(sensitivity)])
        if json.loads(matrix.read_text(encoding="utf-8")).get("contrasts"):
            run_checked([sys.executable, str(HERE / "planned_contrasts.py"),
                         str(matrix), str(canonical), "--out", str(planned_contrasts)])
        if json.loads(matrix.read_text(encoding="utf-8")).get("behavioral_contrasts"):
            run_checked([sys.executable, str(HERE / "behavioral_manipulation_tests.py"),
                         str(matrix), str(canonical), "--out", str(behavioral_tests)])

        provenance = {
            "bundle": args.name,
            "matrix_source": str(matrix.relative_to(ROOT)),
            "record_sources": [str(path.relative_to(ROOT)) for path in record_paths],
            "reference": args.reference,
            "n_question_level_records": len(records),
            "files": {},
        }
        for path in sorted(p for p in stage.rglob("*") if p.is_file()):
            provenance["files"][str(path.relative_to(stage))] = sha256(path)
        (stage / "provenance.json").write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(stage, target)
    print(target)


if __name__ == "__main__":
    main()
