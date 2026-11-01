"""Immutable artifacts and provenance helpers. No network or model imports."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "experiments/results/consensus_recommendation_v1"


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    with Path(path).open(encoding="utf8") as f:
        return [json.loads(line) for line in f if line.strip()]


def create(path, obj, *, rows=False):
    """Create immutable, fsynced artifacts; changed reruns require a new namespace."""
    path = Path(path)
    payload = ("".join(canonical(r) + "\n" for r in obj) if rows
               else json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    if path.exists():
        if path.read_text(encoding="utf8") != payload:
            raise ValueError(f"immutable artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".artifact-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        # A competing creator cannot silently overwrite a completed result.
        try:
            os.link(name, path)
        except FileExistsError:
            if path.read_text(encoding="utf8") != payload:
                raise ValueError(f"concurrent immutable artifact differs: {path}")
    finally:
        os.unlink(name)


def code_hashes():
    return {str(p.relative_to(ROOT)): sha(p)
            for p in sorted(Path(__file__).parent.glob("*.py"))}


def verify_artifacts(base, hashes):
    for name, expected in hashes.items():
        if sha(Path(base) / name) != expected:
            raise ValueError(f"artifact integrity failed: {name}")
