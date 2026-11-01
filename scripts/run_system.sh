#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${MAS_PYTHON:-.venv-native-benchmark/bin/python}"
case "${1:-start}" in
 build) (cd system/frontend/frontend && npm run build) ;;
 start) (cd system/frontend/frontend && npm run build)
   exec "$PYTHON" -m system.application.server --host "${HOST:-127.0.0.1}" --port "${PORT:-8010}" ;;
 lan) (cd system/frontend/frontend && npm run build)
   exec "$PYTHON" -m system.application.server --host 0.0.0.0 --port "${PORT:-8010}" ;;
 test) "$PYTHON" -m unittest system.test_application -v ;;
 *) echo "Usage: bash scripts/run_system.sh {start|lan|build|test}" >&2;exit 2 ;;
esac
