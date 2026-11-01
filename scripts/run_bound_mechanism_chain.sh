#!/usr/bin/env bash
# Two independent diagnostic releases; never mutates paper/probe-v3 runs.
set -euo pipefail
cd "$(dirname "$0")/.."
py="${MAS_PYTHON:-.venv-native-benchmark/bin/python}"
mode="${1:---prepare}"
case "$mode" in
 --prepare)
   bash scripts/run_bound_opportunity_probe.sh --prepare
   "$py" -m experiments.bound_game_bridge prepare ;;
 --run)
   bash scripts/run_bound_opportunity_probe.sh --run
   "$py" -m experiments.bound_game_bridge run --run --max-cny 0.3
   "$py" -m experiments.bound_chain_report ;;
 --report)
   bash scripts/run_bound_opportunity_probe.sh --report
   "$py" -m experiments.bound_game_bridge report
   "$py" -m experiments.bound_chain_report ;;
 *) echo 'Usage: bash scripts/run_bound_mechanism_chain.sh [--prepare|--run|--report]'; exit 2 ;;
esac
