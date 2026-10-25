"""Controlled report protocols for isolating AEIC's measurement value."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


PROTOCOLS = ("free_text", "generic_structure", "aeic")


def protocol_prompt(question: str, protocol: str) -> str:
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown report protocol: {protocol}")
    common = (
        f"Question:\n{question}\n\n"
        "Answer the question using only the knowledge available to you. "
        "Your final answer must begin with Yes or No. "
    )
    if protocol == "free_text":
        return common + "Write a concise analysis followed by the final answer."
    if protocol == "generic_structure":
        return common + (
            'Return strict JSON with exactly two keys: '
            '{"analysis":"concise reasoning","answer":"Yes or No: concise answer"}.'
        )
    return common + (
        'Return strict JSON with exactly four keys: '
        '{"A":"assumptions","E":["specific evidence"],'
        '"I":"inference","C":"Yes or No: concise answer"}.'
    )


def response_format(protocol: str) -> Optional[dict]:
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown report protocol: {protocol}")
    return None if protocol == "free_text" else {"type": "json_object"}


def _json_object(raw: str) -> Dict[str, Any]:
    start = (raw or "").find("{")
    if start < 0:
        raise ValueError("response contains no JSON object")
    value, _ = json.JSONDecoder().raw_decode(raw[start:])
    if not isinstance(value, dict):
        raise ValueError("response JSON must be an object")
    return value


def _stance(text: str) -> Optional[str]:
    matches = re.findall(r"\b(yes|no)\b", text or "", flags=re.IGNORECASE)
    return matches[-1].lower() if matches else None


@dataclass(frozen=True)
class InterfaceObservation:
    protocol: str
    parse_success: bool
    stance: Optional[str]
    stance_extractable: bool
    evidence_items: List[str]
    evidence_boundary_explicit: bool
    fields_present: List[str]
    parse_error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def observe_interface(raw: str, protocol: str) -> InterfaceObservation:
    """Extract only protocol-observable facts; do not judge truth or quality."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown report protocol: {protocol}")
    if protocol == "free_text":
        stance = _stance(raw)
        return InterfaceObservation(
            protocol, bool((raw or "").strip()), stance, stance is not None,
            [], False, [], None if (raw or "").strip() else "empty response",
        )
    try:
        data = _json_object(raw)
        if protocol == "generic_structure":
            required = ("analysis", "answer")
            valid = all(isinstance(data.get(key), str) and data.get(key).strip()
                        for key in required)
            stance = _stance(str(data.get("answer", "")))
            return InterfaceObservation(
                protocol, valid, stance, stance is not None, [], False,
                [key for key in required if key in data],
                None if valid else "missing or invalid generic fields",
            )
        required = ("A", "E", "I", "C")
        evidence = data.get("E")
        valid_evidence = (isinstance(evidence, list)
                          and all(isinstance(item, str) and item.strip() for item in evidence))
        valid = (all(isinstance(data.get(key), str) and data.get(key).strip()
                     for key in ("A", "I", "C")) and valid_evidence)
        stance = _stance(str(data.get("C", "")))
        return InterfaceObservation(
            protocol, valid, stance, stance is not None,
            list(evidence) if valid_evidence else [], valid_evidence,
            [key for key in required if key in data],
            None if valid else "missing or invalid AEIC fields",
        )
    except (ValueError, json.JSONDecodeError) as exc:
        return InterfaceObservation(
            protocol, False, _stance(raw), _stance(raw) is not None,
            [], False, [], f"{type(exc).__name__}: {exc}",
        )

