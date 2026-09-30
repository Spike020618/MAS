"""Atomic, retry-safe JSONL storage for mechanism experiments."""
from __future__ import annotations

import json
import os
import fcntl
from pathlib import Path
from typing import Dict, Iterable, Tuple

from .schema import ExperimentRecord


def record_key(record: ExperimentRecord) -> Tuple[str, str, str, str, int]:
    return (
        record.experiment_id,
        record.dataset,
        record.condition,
        record.question_id,
        record.seed,
    )


class RecordStore:
    """Keeps one latest record per unique experimental unit.

    Error records are retryable.  A successful record atomically replaces an
    older error record with the same key, preventing the duplicate pattern found
    in the legacy online cache.
    """

    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> Dict[Tuple[str, str, str, str, int], dict]:
        records: Dict[Tuple[str, str, str, str, int], dict] = {}
        if not self.path.exists():
            return records
        for line_no, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            item = json.loads(line)
            key = (
                str(item["experiment_id"]), str(item["dataset"]),
                str(item["condition"]), str(item["question_id"]), int(item["seed"]),
            )
            records[key] = item
        return records

    def completed_keys(self) -> set:
        return {
            key for key, value in self.load().items()
            if value.get("completed") is True and not value.get("error")
        }

    def upsert(self, record: ExperimentRecord) -> None:
        errors = record.validate(require_complete=record.completed)
        if errors:
            raise ValueError("invalid experiment record: " + "; ".join(errors))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            current = self.load()
            key = record_key(record)
            previous = current.get(key)
            if previous:
                # A second worker may have started before the first worker
                # committed.  Once a unit is complete it is immutable: a
                # concurrent success is not another experimental attempt.
                if previous.get("completed") is True and not previous.get("error"):
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                    return
                record.attempt_count = int(previous.get("attempt_count", 1)) + 1
                prior = list(previous.get("prior_errors") or [])
                if previous.get("error"):
                    prior.append(str(previous["error"]))
                record.prior_errors = prior
            current[key] = record.to_dict()
            self._atomic_write(current.values())
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _atomic_write(self, records: Iterable[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            for item in records:
                handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self.path)
