#!/usr/bin/env bash
# Resume the interrupted paper-v2 main comparison in a NEW root, replaying exact 200 receipts.
# Old root stays untouched (interrupted record). Denied 400 slots are re-dispatched after recharge;
# 58 unknown intents are preserved without resend (recorded as invalid events, matching old semantics).
set -euo pipefail
cd "$(dirname "$0")/.."
py="${MAS_PYTHON:-.venv-native-benchmark/bin/python}"
src="experiments/results/consensus_mechanism_paper_v2"
dst="experiments/results/consensus_mechanism_paper_v2r1"
mode="${1:---prepare}"
if [[ "$mode" != --prepare && "$mode" != --run ]]; then echo 'Usage: bash scripts/resume_consensus_paper_v2r1.sh [--prepare|--run]'; exit 2; fi
for f in data/train.json data/test.json model/model.joblib model/training_tasks.json; do
  if [[ ! -f "$src/$f" ]]; then echo "missing source file: $src/$f"; exit 1; fi
done
if [[ "$mode" == --prepare ]]; then
  mkdir -p "$dst/data" "$dst/model"
  for f in data/train.json data/test.json model/model.joblib model/training_tasks.json; do
    if [[ -f "$dst/$f" ]]; then cmp -s "$src/$f" "$dst/$f" || { echo "existing $dst/$f differs from source; resolve manually"; exit 1; }
    else cp "$src/$f" "$dst/$f"; fi
  done
  exec "$py" -m experiments.consensus_mechanism.cli main --root "$dst" --max-cny 5
else
  # Requires account in good standing; the account guard stops the batch fail-closed on arrears.
  exec "$py" -m experiments.consensus_mechanism.cli main --root "$dst" --run --max-cny 5 --reuse "$src/main"
fi
