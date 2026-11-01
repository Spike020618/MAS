#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${MAS_PYTHON:-.venv-native-benchmark/bin/python}"
if [[ $# -lt 2 ]]; then
  echo 'Usage: bash scripts/run_mechanism_suite.sh INPUTS_JSON NEW_OUTPUT_ROOT [--run]'
  echo 'Default: prepare only, no model calls. --run starts paid generation.'
  exit 2
fi
inputs="$1"
output="$2"
shift 2
exec "$PY" -m experiments.evidence_consensus.runner_general --inputs "$inputs" --output "$output" "$@"
