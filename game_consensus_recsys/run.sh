#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

MODE="${1:-demo}"

case "$MODE" in
  doctor)
    python3 -m game_consensus_recsys.doctor
    ;;
  demo)
    python3 -m game_consensus_recsys.doctor
    python3 -m game_consensus_recsys.demo
    ;;
  api)
    python3 -m game_consensus_recsys.http_server --host 127.0.0.1 --port "${PORT:-8000}"
    ;;
  test)
    python3 -m unittest tests.test_deepseek_recsys tests.test_game_consensus_recsys tests.test_mechanism
    ;;
  *)
    echo "Usage: $0 {doctor|demo|api|test}" >&2
    exit 2
    ;;
esac
