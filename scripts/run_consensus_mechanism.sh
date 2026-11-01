#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
py="${MAS_PYTHON:-.venv-native-benchmark/bin/python}"
root="${1:-experiments/results/consensus_mechanism_set_v2r1}"
stage="${2:-prepare}"
if [[ $# -ge 2 ]]; then shift 2; elif [[ $# -eq 1 ]]; then shift; fi
exec "$py" -m experiments.consensus_mechanism.cli "$stage" --root "$root" "$@"
