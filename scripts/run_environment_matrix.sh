#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -lt 2 ]]; then
  echo 'Usage: bash scripts/run_environment_matrix.sh SUITE_ROOT NEW_RUNS_ROOT [--estimator heuristic|learned|perturbed] [--model PATH] [--run]'
  echo 'Without --run: preparation only, no model calls.'
  exit 2
fi
suite="$1"
runs="$2"
shift 2
for environment in baseline missing redundant distractor; do
  bash scripts/run_mechanism_suite.sh "$suite/$environment/tasks.json" "$runs/$environment" "$@"
done
