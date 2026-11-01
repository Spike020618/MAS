#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
py="${MAS_PYTHON:-.venv-native-benchmark/bin/python}"
mode="${1:---prepare}"
case "$mode" in
 --prepare) exec "$py" -m experiments.calibrated_action_value prepare ;;
 --run)
   "$py" - <<'PY'
import json
from pathlib import Path
r=Path('experiments/results/calibrated_action_value_v2/train/cases')
paths=[r/f'case_{i}.json'for i in range(4)]
if all(p.exists()for p in paths):
    valid=sum(a['event']['valid']for p in paths for a in json.loads(p.read_text())['branches'].values())
    if valid<13:raise RuntimeError('Existing first-four engineering gate failed; do not bypass on resume')
PY
   "$py" -m experiments.calibrated_action_value train --run --max-cny 1
   "$py" -m experiments.calibrated_action_value fit --run --max-cny 1
   "$py" -m experiments.run_same_data_ridge
   "$py" -m experiments.calibrated_action_value evaluate --run --max-cny 1
   "$py" -m experiments.run_same_data_ridge --run
   "$py" -m experiments.calibrated_value_report panel --run
   "$py" -m experiments.calibrated_value_report report ;;
 --followup)
   "$py" -m experiments.run_value_calibration_followup
   "$py" -m experiments.run_state_union_control
   "$py" -m experiments.run_state_union_control --model-family ridge
   "$py" -m experiments.run_value_calibration_followup --run
   "$py" -m experiments.run_state_union_control --run
   "$py" -m experiments.run_state_union_control --model-family ridge --run
   "$py" -m experiments.calibrated_value_report panel --root experiments/results/calibrated_action_value_v3 --run
   "$py" -m experiments.final_action_value_report ;;
 --report)
   "$py" -m experiments.calibrated_value_report report
   "$py" -m experiments.final_action_value_report ;;
 *) echo 'Usage: bash scripts/run_calibrated_action_value.sh [--prepare|--run|--followup|--report]'; exit 2 ;;
esac
