#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
case "${1:-check}" in
 test) .venv-native-benchmark/bin/python -m unittest experiments.tests.test_coordination_game_v1 -v ;;
 check|estimate|exposure) .venv-native-benchmark/bin/python -m experiments.coordination_game_v1.cli "$@" ;;
 *) echo 'P0 only: check | estimate | exposure | test. P1 collection is not enabled.' >&2;exit 2 ;;
esac
