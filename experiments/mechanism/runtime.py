"""Shared resume and retry policy for long-running mechanism experiments."""
from __future__ import annotations

import re
from typing import Iterable, Mapping


TRANSIENT_PATTERNS = (
    r"connection(error|reset|aborted)?",
    r"timeout(error)?",
    r"timed out",
    r"temporary failure",
    r"name resolution",
    r"remote end closed",
    r"server disconnected",
    r"http (408|409|425|429|5\d\d)\b",
)


def classify_error(error: str | None) -> str:
    """Return transient, data, protocol, or unknown without exposing secrets."""
    text = (error or "").strip().lower()
    if not text:
        return "none"
    if any(re.search(pattern, text) for pattern in TRANSIENT_PATTERNS):
        return "transient"
    if any(marker in text for marker in (
        "no aeic json", "jsondecode", "must be", "invalid experiment record",
        "question ids not found", "local question cache missing",
    )):
        return "data"
    if any(marker in text for marker in (
        "http 400", "http 401", "http 403", "http 404", "unsupported",
        "api key required", "invalid_request", "permission",
    )):
        return "protocol"
    return "unknown"


def contiguous_resume_prefix(question_ids: Iterable[str], completed: Mapping[str, dict]) -> list[str]:
    """Return the completed prefix; reject future checkpoints after a gap.

    Longitudinal reputation can only be reconstructed from a contiguous prefix.
    Silently resuming a missing earlier question from a later reputation snapshot
    would contaminate every subsequent unit.
    """
    ordered = [str(x) for x in question_ids]
    prefix: list[str] = []
    gap_seen = False
    future: list[str] = []
    for qid in ordered:
        done = qid in completed
        if done and not gap_seen:
            prefix.append(qid)
        elif not done:
            gap_seen = True
        elif done:
            future.append(qid)
    if future:
        raise RuntimeError(
            "non-contiguous checkpoint: completed units exist after the first gap: "
            + ", ".join(future[:10])
        )
    return prefix
