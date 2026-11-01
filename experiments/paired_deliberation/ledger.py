"""Experiment-level accounting shared by every paired arm."""
from decimal import Decimal, InvalidOperation


STAGES = {"initial_generation", "review", "aggregation", "tool"}


def _integer(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("invalid " + name)
    return value


def _money(value):
    if value is None:
        return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError("invalid actual_cost") from None
    if not parsed.is_finite() or parsed < 0:
        raise ValueError("invalid actual_cost")
    return parsed


class ExperimentLedger:
    """Append-only accounting; it never estimates missing provider usage as zero."""

    def __init__(self):
        self.records = []

    def add(self, stage, prompt_tokens, completion_tokens, tool_calls,
            tool_units, retries, wall_time_ms, actual_cost, metadata=None):
        if stage not in STAGES:
            raise ValueError("unknown accounting stage")
        if not isinstance(metadata, dict):
            raise ValueError("metadata is required")
        row = {
            "stage": stage,
            "prompt_tokens": _integer(prompt_tokens, "prompt_tokens"),
            "completion_tokens": _integer(completion_tokens, "completion_tokens"),
            "tool_calls": _integer(tool_calls, "tool_calls"),
            "tool_units": _integer(tool_units, "tool_units"),
            "retries": _integer(retries, "retries"),
            "wall_time_ms": _integer(wall_time_ms, "wall_time_ms"),
            "actual_cost": (None if _money(actual_cost) is None else str(_money(actual_cost))),
            "metadata": dict(metadata),
        }
        self.records.append(row)
        return dict(row)

    def summary(self):
        if not self.records:
            raise ValueError("empty ledger cannot certify a run")
        keys = (
            "prompt_tokens", "completion_tokens", "tool_calls", "tool_units",
            "retries", "wall_time_ms",
        )
        totals = {key: sum(row[key] for row in self.records) for key in keys}
        known_costs = [row["actual_cost"] for row in self.records if row["actual_cost"] is not None]
        totals["actual_cost_complete"] = len(known_costs) == len(self.records)
        totals["actual_cost"] = (str(sum(Decimal(value) for value in known_costs))
                                 if totals["actual_cost_complete"] else None)
        totals["records"] = len(self.records)
        totals["stages"] = sorted({row["stage"] for row in self.records})
        return totals

    def assert_complete_path(self, require_review=True):
        present = {row["stage"] for row in self.records}
        required = {"initial_generation", "aggregation"}
        if require_review:
            required.add("review")
        if not required <= present:
            raise ValueError("path ledger is missing required stages")
        return self.summary()
