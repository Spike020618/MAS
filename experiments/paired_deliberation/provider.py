"""Checkpointed provider boundary; inert unless explicitly authorized."""
import json
from pathlib import Path

from experiments.consensus_recommendation.common import create, fingerprint


class CheckpointedProvider:
    """Wrap an injected transport and refuse ambiguous retries or stale reuse."""

    def __init__(self, root, plan_hash, transport, authorized=False):
        self.root = Path(root)
        self.plan_hash = plan_hash
        self.transport = transport
        self.authorized = authorized

    def complete(self, slot, request):
        if not isinstance(slot, str) or not slot or "/" in slot or ".." in slot:
            raise ValueError("unsafe slot identity")
        dependency = fingerprint({"plan_hash": self.plan_hash, "slot": slot, "request": request})
        path = self.root / (slot + ".json")
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["dependency_hash"] != dependency or saved["integrity"] != fingerprint(saved["record"]):
                raise ValueError("stale or tampered provider checkpoint: " + str(path))
            return saved["record"]
        intent = self.root / (slot + ".intent.json")
        if intent.exists():
            raise RuntimeError("unfinished request has unknown billing state")
        if not self.authorized:
            raise RuntimeError("provider execution is not authorized")
        create(intent, {"dependency_hash": dependency, "slot": slot})
        record = self.transport(copy_json(request))
        validate_provider_record(record)
        create(path, {"dependency_hash": dependency, "record": record,
                      "integrity": fingerprint(record)})
        intent.unlink()
        return record


def copy_json(value):
    return json.loads(json.dumps(value, ensure_ascii=False))


def validate_provider_record(record):
    allowed = {"report", "usage", "model", "elapsed_ms", "integrity_issues"}
    if not isinstance(record, dict) or set(record) - allowed or not {"report", "usage", "model", "elapsed_ms"} <= set(record):
        raise ValueError("provider record schema")
    if not isinstance(record.get("integrity_issues", []), list):
        raise ValueError("provider integrity issues schema")
    usage = record["usage"]
    if not isinstance(usage, dict) or set(usage) != {"prompt_tokens", "completion_tokens", "tool_calls",
                                                      "tool_units", "actual_cost", "retries"}:
        raise ValueError("provider usage schema")
    for key in ("prompt_tokens", "completion_tokens", "tool_calls", "tool_units", "retries"):
        if not isinstance(usage[key], int) or isinstance(usage[key], bool) or usage[key] < 0:
            raise ValueError("invalid provider usage")
    if not isinstance(record["elapsed_ms"], int) or record["elapsed_ms"] < 0:
        raise ValueError("invalid provider elapsed time")
    if not isinstance(record["model"], str) or not record["model"]:
        raise ValueError("missing returned model identity")
