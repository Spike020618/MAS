"""Raw HTTP boundary, parser, and complete request accounting for V5."""
import json
import os
import time
from urllib.parse import urlparse


class RawDeepSeek:
    def __init__(self, model="deepseek-chat", api_key=None, base_url="https://api.deepseek.com"):
        parsed = urlparse(base_url)
        if parsed.scheme != "https" or parsed.netloc != "api.deepseek.com" or parsed.path not in {"", "/v1"}:
            raise ValueError("unsupported DeepSeek endpoint")
        self.url = base_url.rstrip("/") + ("/chat/completions" if parsed.path == "/v1" else "/v1/chat/completions")
        self.model = model
        self.key = api_key if api_key is not None else os.environ.get("DEEPSEEK_API_KEY", "")

    def preflight(self, request):
        if not self.key:
            raise RuntimeError("API key missing; request intent not created")
        if request.get("labels_available") is not False:
            raise ValueError("label isolation flag missing")

    def __call__(self, request):
        self.preflight(request)
        import requests
        started = time.monotonic()
        response = requests.post(self.url, headers={"Authorization": "Bearer " + self.key},
            json={"model": self.model, "messages": request["messages"], "temperature": 0,
                  "max_tokens": request["max_tokens"], "response_format": {"type": "json_object"},
                  "stream": False}, timeout=(15, 150))
        # All response bytes, including HTTP and parse failures, precede parsing.
        return {"status_code": response.status_code, "body": response.text,
                "elapsed_ms": int((time.monotonic() - started) * 1000)}


def parse_raw(raw, request):
    if raw["status_code"] != 200:
        raise ValueError("HTTP error persisted; no automatic network retry")
    obj = json.loads(raw["body"])
    choice = obj["choices"][0]
    if choice.get("finish_reason") != "stop":
        raise ValueError("truncated response persisted")
    report = json.loads(choice["message"]["content"])
    if not isinstance(report, dict):
        raise ValueError("expected JSON object")
    # Provider envelope metadata, applied identically to all response kinds.
    # Never coerce or repair verdicts, targets, citations, claims or rankings.
    if report.get("type") == "json_object":
        report = {key: value for key, value in report.items() if key != "type"}
    usage = obj.get("usage", {})
    for key in ("prompt_tokens", "completion_tokens"):
        if type(usage.get(key)) is not int or usage[key] < 0:
            raise ValueError("unknown usage persisted")
    if not isinstance(obj.get("model"), str) or not obj["model"]:
        raise ValueError("missing returned model")
    return {"report": report, "usage": {"prompt_tokens": usage["prompt_tokens"],
             "completion_tokens": usage["completion_tokens"], "tool_calls": 0,
             "tool_units": 0, "retries": 0, "actual_cost": None},
            "model": obj["model"], "elapsed_ms": raw["elapsed_ms"]}


class BoundedClient:
    def __init__(self, provider, limits):
        self.provider, self.limits = provider, dict(limits)
        self.records = []
        self.input_upper = 0
        self.output_reserved = 0
        self.model = None

    def complete(self, slot, request):
        self._reserve(request)
        record = self.provider.complete(slot, request)
        self._account(slot, request, record)
        return record

    def _reserve(self, request):
        upper = sum(len(message["content"].encode("utf8")) + 512 for message in request["messages"])
        cap = request["max_tokens"]
        if type(cap) is not int or cap <= 0 or cap > self.limits["per_call_completion_tokens"]:
            raise ValueError("invalid output reservation")
        if upper > self.limits["per_call_prompt_tokens"]:
            raise ValueError("input upper bound exceeds per-call budget before dispatch")
        if self.input_upper + upper > self.limits["total_prompt_tokens"]:
            raise ValueError("input upper bound exceeds path budget before dispatch")
        if self.output_reserved + cap > self.limits["total_completion_tokens"]:
            raise ValueError("output reservation exceeds path budget before dispatch")
        self.input_upper += upper
        self.output_reserved += cap

    def account_shared(self, slot, request, record):
        """Charge a shared initial/feedback call to this logical path once."""
        if any(row["slot"] == slot for row in self.records):
            raise ValueError("shared call already charged")
        self._reserve(request)
        self._account(slot, request, record, shared=True)

    def _account(self, slot, request, record, shared=False):
        self.records.append({"slot": slot, "kind": request["kind"], "logical_shared": shared, **record})
        if self.model is not None and self.model != record["model"]:
            raise ValueError("returned model changed within path")
        self.model = record["model"]

    def summary(self):
        return {"calls": len(self.records), "model": self.model,
                "prompt_tokens": sum(row["usage"]["prompt_tokens"] for row in self.records),
                "completion_tokens": sum(row["usage"]["completion_tokens"] for row in self.records),
                "input_utf8_upper_reserved": self.input_upper,
                "output_tokens_reserved": self.output_reserved,
                "cumulative_request_ms": sum(row["elapsed_ms"] for row in self.records),
                "actual_cost": None}
