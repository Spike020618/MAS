#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -lt 2 ]]; then echo 'Usage: bash scripts/run_protocol_v2.sh INPUTS NEW_OUTPUT [--estimator heuristic|learned --model PATH] [--run]'; exit 2; fi
inputs="$1"; output="$2"; shift 2
exec "${MAS_PYTHON:-.venv-native-benchmark/bin/python}" -m experiments.evidence_consensus.runner_protocol_v2 --inputs "$inputs" --output "$output" "$@"
