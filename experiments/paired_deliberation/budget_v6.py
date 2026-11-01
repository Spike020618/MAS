"""Attempt accounting and conservative pre-dispatch compound reservations."""
import copy
import json
from pathlib import Path


class BudgetExceeded(ValueError):
    pass


class AttemptClient:
    def __init__(self, provider, limits):
        self.provider, self.limits = provider, copy.deepcopy(limits)
        self.records, self.reserved_prompt, self.reserved_output = [], 0, 0
        self.model = None

    @staticmethod
    def upper(request):
        return sum(len(m["content"].encode("utf8")) + 512 for m in request["messages"])

    def afford(self, calls, prompt, completion):
        return (len(self.records) + calls <= self.limits["max_calls"]
                and self.reserved_prompt + prompt <= self.limits["total_prompt_tokens"]
                and self.reserved_output + completion <= self.limits["total_completion_tokens"])

    def action_fits(self, calls, reserve_final=True):
        n = calls + int(reserve_final)
        return self.afford(n, n * self.limits["per_call_prompt_tokens"],
                           n * self.limits["per_call_completion_tokens"])

    def _reserve(self, slot, request, shared=False):
        if any(row["slot"] == slot for row in self.records):
            raise ValueError("duplicate logical request slot")
        prompt, output = self.upper(request), request["max_tokens"]
        if (request.get("labels_available") is not False or type(output) is not int or output <= 0
                or output > self.limits["per_call_completion_tokens"]):
            raise ValueError("invalid request reservation")
        if prompt > self.limits["per_call_prompt_tokens"] or not self.afford(1, prompt, output):
            raise BudgetExceeded("request exceeds path budget before dispatch")
        self.reserved_prompt += prompt
        self.reserved_output += output
        row = {"slot": slot, "kind": request["kind"], "shared": shared,
               "prompt_upper": prompt, "completion_reserved": output, "status": "attempted",
               "usage": None, "model": None, "elapsed_ms": None}
        self.records.append(row)
        return row

    def _finish(self, row, record):
        row.update(status="success", usage=copy.deepcopy(record["usage"]), model=record["model"],
                   elapsed_ms=record["elapsed_ms"])
        if (any(type(row["usage"].get(k)) is not int or row["usage"][k] < 0
                for k in ("prompt_tokens", "completion_tokens"))
                or row["usage"]["prompt_tokens"] > row["prompt_upper"]
                or row["usage"]["completion_tokens"] > row["completion_reserved"]):
            row["status"] = "usage_contract_failure"
            raise ValueError("provider usage invalid or exceeded reservation")
        if self.model is not None and self.model != record["model"]:
            row["status"] = "model_mismatch"
            raise ValueError("returned model changed within path")
        self.model = record["model"]

    def complete(self, slot, request):
        row = self._reserve(slot, request)
        try:
            record = self.provider.complete(slot, request)
        except Exception as exc:
            row.update(status="failed", error=type(exc).__name__ + ": " + str(exc))
            root = getattr(self.provider, "root", None)
            raw_path = Path(root) / (slot + ".raw.json") if root is not None else None
            if raw_path is not None and raw_path.exists():
                raw = json.loads(raw_path.read_text())["raw"]
                row["elapsed_ms"] = raw.get("elapsed_ms")
                try:
                    body = json.loads(raw["body"])
                    usage = body["usage"]
                    if all(type(usage.get(k)) is int and usage[k] >= 0 for k in ("prompt_tokens", "completion_tokens")):
                        row["usage"] = {k: usage[k] for k in ("prompt_tokens", "completion_tokens")}
                    row["model"] = body.get("model")
                except (ValueError, KeyError, TypeError):
                    pass
            raise
        self._finish(row, record)
        return record

    def account_shared(self, slot, request, record):
        self._finish(self._reserve(slot, request, shared=True), record)

    def summary(self):
        complete = all(row["usage"] is not None for row in self.records)
        def total(key):
            known = sum(row["usage"][key] for row in self.records if row["usage"] is not None)
            return {"known": known, "total": known if complete else None}
        return {"attempts": len(self.records), "successes": sum(r["status"] == "success" for r in self.records),
                "usage_complete": complete, "prompt_tokens": total("prompt_tokens"),
                "completion_tokens": total("completion_tokens"), "model": self.model,
                "reserved_prompt_upper": self.reserved_prompt, "reserved_output": self.reserved_output,
                "actual_currency_cost": None, "records": copy.deepcopy(self.records)}
