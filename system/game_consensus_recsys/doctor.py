"""Print a safe runtime readiness report without exposing credentials."""
from __future__ import annotations

import importlib.util
import json
import platform

from .deepseek import DeepSeekConfig


def report():
    config = DeepSeekConfig.from_environment()
    return {
        "python": platform.python_version(),
        "mode": "deepseek" if config else "offline_deterministic",
        "deepseek": {
            "configured": bool(config),
            "base_url": config.base_url if config else None,
            "model": config.model if config else None,
            "api_key_exposed": False,
        },
        "optional_api_runtime": {
            "fastapi": importlib.util.find_spec("fastapi") is not None,
            "uvicorn": importlib.util.find_spec("uvicorn") is not None,
        },
        "capabilities": {
            "demo": True,
            "deterministic_fallback": True,
            "deepseek_logical_agents": bool(config),
            "http_api": True,
            "fastapi_adapter": all(importlib.util.find_spec(name) is not None for name in ("fastapi", "uvicorn")),
        },
    }


def main():
    print(json.dumps(report(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
