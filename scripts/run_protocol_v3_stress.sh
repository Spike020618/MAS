#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
py="${MAS_PYTHON:-.venv-native-benchmark/bin/python}"
root="${1:-experiments/results/protocol_v3_frozen}"
mode="${2:---prepare}"
if [[ "$mode" != --prepare && "$mode" != --run ]]; then echo 'Usage: bash scripts/run_protocol_v3_stress.sh ROOT [--prepare|--run]'; exit 2; fi
"$py" -m experiments.prepare_protocol_v3_stress --inputs "$root/inputs/diagnostic/tasks.json" --original-inputs experiments/results/e2_action_acquisition_160_live_v1/inputs.json --output "$root/stress_inputs"
if [[ "$mode" == --prepare ]]; then echo 'Prepared three stress inputs, no API calls.'; exit 0; fi
for environment in missing redundant distractor; do
 "$py" -m experiments.evidence_consensus.runner_protocol_v3 --inputs "$root/stress_inputs/$environment/tasks.json" --output "$root/stress/$environment" --estimator learned --model "$root/model/model.joblib" --run
done

"$py" -m experiments.report_protocol_v3 --root "$root" --stress-only
