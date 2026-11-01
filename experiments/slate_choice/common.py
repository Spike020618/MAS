from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "experiments/results/slate_choice_v1"
PROTOCOL = Path(__file__).with_name("protocol.json")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_once(path, value, jsonl=False):
    path = Path(path)
    content = ("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in value)
               if jsonl else json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"different output requires new version: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def rows(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def user_bucket(user_id):
    return int(hashlib.sha256(f"slate-choice-user-v1:{user_id}".encode()).hexdigest(), 16) % 10


def sample_key(user_id):
    return hashlib.sha256(f"slate-choice-sample-v1:{user_id}".encode()).hexdigest()
