#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
py="${MAS_PYTHON:-.venv-native-benchmark/bin/python}"
root="${1:-experiments/results/protocol_v2_frozen}"
mode="${2:---prepare}"
if [[ "$mode" != --prepare && "$mode" != --run ]]; then echo 'Usage: bash scripts/run_protocol_v2_pipeline.sh [NEW_ROOT] [--prepare|--run]'; exit 2; fi
"$py" -m experiments.prepare_disjoint_acquisition --inputs experiments/results/e2_action_acquisition_160_live_v1/inputs.json --output "$root/inputs"
"$py" -m experiments.select_protocol_partition --inputs "$root/inputs/tasks.json" --output "$root/diagnostic" --partition diagnostic
"$py" -m experiments.acquire_disjoint_actions --inputs "$root/inputs/tasks.json" --output "$root/acquisition" ${mode/--prepare/}
if [[ "$mode" != --run ]]; then
 echo 'Preparation passed; no API calls. Repeat with --run to collect, fit and compare.'
 exit 0
fi
"$py" -m experiments.fit_disjoint_contribution --run-root "$root/acquisition" --output "$root/model" --fit
bash scripts/run_protocol_v2.sh "$root/diagnostic/tasks.json" "$root/heuristic" --estimator heuristic --run
bash scripts/run_protocol_v2.sh "$root/diagnostic/tasks.json" "$root/learned" --estimator learned --model "$root/model/model.joblib" --run
echo "Completed. Results: $root/heuristic and $root/learned"
