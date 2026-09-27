"""Minimal DeepSeek JSON client using only the Python standard library."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class DeepSeekError(RuntimeError):
    pass


def _load_env_file(path: Path) -> None:
    """Load a simple dotenv file without replacing exported environment values."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if value[:1] == value[-1:] and value[:1] in {'"', "'"}:
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


def _load_project_env() -> None:
    _load_env_file(Path(__file__).resolve().parent.parent / ".env")


@dataclass(frozen=True)
class DeepSeekConfig:
    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-flash"
    timeout_s: float = 30.0
    max_retries: int = 2
    max_tokens: int = 2400

    @classmethod
    def from_environment(cls, load_project_env: bool = True) -> Optional["DeepSeekConfig"]:
        if load_project_env:
            _load_project_env()
        key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not key:
            return None
        return cls(
            api_key=key,
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/"),
            model=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            timeout_s=float(os.environ.get("DEEPSEEK_TIMEOUT_S", "30")),
            max_retries=int(os.environ.get("DEEPSEEK_MAX_RETRIES", "2")),
            max_tokens=int(os.environ.get("DEEPSEEK_MAX_TOKENS", "2400")),
        )


class DeepSeekClient:
    def __init__(
        self,
        config: DeepSeekConfig,
        transport: Optional[Callable[[urllib.request.Request, float], bytes]] = None,
    ):
        self.config = config
        self._transport = transport or self._urlopen

    @staticmethod
    def _urlopen(request: urllib.request.Request, timeout: float) -> bytes:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.config.base_url}/chat/completions",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        last_error: Optional[Exception] = None
        for attempt in range(self.config.max_retries + 1):
            try:
                raw = self._transport(request, self.config.timeout_s)
                response = json.loads(raw.decode("utf-8"))
                content = response["choices"][0]["message"]["content"]
                if not content:
                    raise DeepSeekError("DeepSeek returned empty JSON content")
                result = json.loads(content)
                result["_usage"] = response.get("usage") or {}
                result["_model"] = response.get("model") or self.config.model
                return result
            except (KeyError, IndexError, json.JSONDecodeError, UnicodeDecodeError,
                    urllib.error.URLError, TimeoutError, DeepSeekError) as error:
                last_error = error
                if attempt < self.config.max_retries:
                    time.sleep(min(0.25 * (2 ** attempt), 1.0))
        raise DeepSeekError(f"DeepSeek request failed after retries: {last_error}") from last_error
