#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
py="${MAS_PYTHON:-.venv-native-benchmark/bin/python}"
root="${1:-experiments/results/protocol_v3_frozen}"
mode="${2:---prepare}"
if [[ "$mode" != --prepare && "$mode" != --probe && "$mode" != --run ]]; then echo 'Usage: bash scripts/run_protocol_v3_pipeline.sh [NEW_ROOT] [--prepare|--probe|--run]'; exit 2; fi
"$py" -m experiments.prepare_protocol_v3 --inputs experiments/results/e2_action_acquisition_160_live_v1/inputs.json --output "$root/inputs"
"$py" -m experiments.acquire_protocol_v3 --inputs "$root/inputs/training/tasks.json" --output "$root/acquisition"
if [[ "$mode" == --prepare ]]; then echo 'Prepared, no API calls. --probe checks six training tasks; --run executes the full pipeline after that gate.'; exit 0; fi
if [[ "$mode" == --probe ]]; then
 "$py" -m experiments.acquire_protocol_v3 --inputs "$root/inputs/training/tasks.json" --output "$root/acquisition" --run --probe-only
 exit 0
fi
"$py" -m experiments.acquire_protocol_v3 --inputs "$root/inputs/training/tasks.json" --output "$root/acquisition" --run
"$py" -m experiments.fit_protocol_v3 --run-root "$root/acquisition" --output "$root/model"
"$py" -m experiments.evidence_consensus.runner_protocol_v3 --inputs "$root/inputs/diagnostic/tasks.json" --output "$root/heuristic" --estimator heuristic --run
"$py" -m experiments.evidence_consensus.runner_protocol_v3 --inputs "$root/inputs/diagnostic/tasks.json" --output "$root/learned" --estimator learned --model "$root/model/model.joblib" --run
"$py" -m experiments.report_protocol_v3 --root "$root"
echo "Completed: $root/report.md"
