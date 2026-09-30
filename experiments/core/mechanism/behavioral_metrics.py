"""Deterministic process metrics for longitudinal AEIC reports."""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List, Optional, Sequence, Set

from .schema import AEICReport


_TOKEN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def _tokens(text: str) -> Set[str]:
    return {m.group(0).lower() for m in _TOKEN.finditer(text or "")}


def jaccard_text(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def evidence_novelty(report: AEICReport, visible_evidence: Iterable[str]) -> float:
    """Fraction of evidence tokens not already visible to the agent."""
    own = _tokens(" ".join(report.E))
    prior = _tokens(" ".join(visible_evidence))
    if not own:
        return 0.0
    return len(own - prior) / len(own)


def evidence_copy_score(report: AEICReport, visible_evidence: Iterable[str]) -> float:
    visible = list(visible_evidence)
    if not report.E or not visible:
        return 0.0
    return sum(max(jaccard_text(e, p) for p in visible) for e in report.E) / len(report.E)


def stance_flip(previous: Optional[str], current: Optional[str]) -> bool:
    return bool(previous and current and previous != current)


def unsupported_flip(
    previous_report: AEICReport,
    current_report: AEICReport,
    previous_stance: Optional[str],
    current_stance: Optional[str],
    new_external_evidence: Sequence[str],
    evidence_change_threshold: float = 0.25,
) -> bool:
    """A stance changed without meaningful internal or external evidence change."""
    if not stance_flip(previous_stance, current_stance):
        return False
    evidence_similarity = jaccard_text(
        " ".join(previous_report.E), " ".join(current_report.E)
    )
    meaningful_internal_change = (1.0 - evidence_similarity) >= evidence_change_threshold
    return not meaningful_internal_change and not any(x.strip() for x in new_external_evidence)


def majority_stance(stances: Sequence[Optional[str]]) -> Optional[str]:
    values = [s for s in stances if s]
    if not values:
        return None
    counts = Counter(values)
    top = counts.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return None
    return top[0][0]

