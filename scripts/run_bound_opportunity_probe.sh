#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
py="${MAS_PYTHON:-.venv-native-benchmark/bin/python}"
mode="${1:---prepare}"
root="experiments/results/bound_opportunity_v3"
case "$mode" in
 --prepare) exec "$py" -m experiments.bound_opportunity_probe prepare --root "$root" ;;
 --run) exec "$py" -m experiments.bound_opportunity_probe run --root "$root" --run --max-cny 0.2 ;;
 --report) exec "$py" -m experiments.bound_opportunity_probe report --root "$root" ;;
 --budget-check) exec "$py" -m experiments.bound_opportunity_probe budget_check --root "$root" ;;
 *) echo 'Usage: bash scripts/run_bound_opportunity_probe.sh [--prepare|--run|--report|--budget-check]'; exit 2 ;;
esac
