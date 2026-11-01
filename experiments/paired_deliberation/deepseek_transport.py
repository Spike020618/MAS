"""Official DeepSeek transport for a separately authorized experiment run."""
import json
import os
import time
from urllib.parse import urlparse

from experiments.consensus_recommendation.protocol import collect_parse


class DeepSeekTransport:
    def __init__(self, base_url=None, model=None, api_key=None, timeout=(15, 150)):
        base = (base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip("/")
        parsed = urlparse(base)
        if parsed.scheme != "https" or parsed.hostname != "api.deepseek.com":
            raise ValueError("only the official scoped DeepSeek endpoint is accepted")
        self.url = base + ("/chat/completions" if base.endswith("/v1") else "/v1/chat/completions")
        self.model = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
        self.api_key = api_key if api_key is not None else os.environ.get("DEEPSEEK_API_KEY", "")
        self.timeout = timeout

    def __call__(self, request):
        if not self.api_key:
            raise RuntimeError("DeepSeek API key is absent; no request sent")
        if request.get("labels_available") is not False:
            raise ValueError("execution request did not certify label isolation")
        required = {"messages", "evidence", "prior", "max_tokens", "labels_available"}
        if not required <= set(request):
            raise ValueError("incomplete transport request")
        import requests
        started = time.perf_counter()
        response = requests.post(
            self.url,
            headers={"Authorization": "Bearer " + self.api_key},
            json={"model": self.model, "messages": request["messages"], "temperature": 0,
                  "max_tokens": request["max_tokens"], "stream": False,
                  "response_format": {"type": "json_object"}},
            timeout=self.timeout,
        )
        elapsed = int(round((time.perf_counter() - started) * 1000))
        if response.status_code != 200:
            raise RuntimeError("provider HTTP %d; no automatic retry" % response.status_code)
        raw = response.json()
        choice = raw["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise ValueError("provider response did not finish cleanly")
        report, issues = collect_parse(
            choice["message"]["content"], request["evidence"], request.get("prior")
        )
        usage = raw.get("usage")
        if not isinstance(usage, dict):
            raise ValueError("provider usage missing")
        return {
            "report": report,
            "usage": {"prompt_tokens": usage["prompt_tokens"],
                      "completion_tokens": usage["completion_tokens"],
                      "tool_calls": 0, "tool_units": 0,
                      "actual_cost": None, "retries": 0},
            "model": raw.get("model") or self.model,
            "elapsed_ms": elapsed,
            "integrity_issues": issues,
        }
