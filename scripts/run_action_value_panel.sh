#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
py="${MAS_PYTHON:-.venv-native-benchmark/bin/python}"
root="${1:-experiments/results/action_value_panel_v1}"
mode="${2:---prepare}"
if [[ "$mode" != --prepare && "$mode" != --run ]]; then echo 'Usage: bash scripts/run_action_value_panel.sh [NEW_ROOT] [--prepare|--run] [extra CLI args]'; exit 2; fi
if [[ $# -ge 2 ]]; then shift 2; elif [[ $# -eq 1 ]]; then shift; fi
if [[ "$mode" == --run ]]; then
 exec "$py" -m experiments.action_value_panel --inputs experiments/results/protocol_v3_frozen/inputs/diagnostic/tasks.json --output "$root" --run "$@"
else
 exec "$py" -m experiments.action_value_panel --inputs experiments/results/protocol_v3_frozen/inputs/diagnostic/tasks.json --output "$root" "$@"
fi
