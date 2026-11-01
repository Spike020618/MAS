#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
py="${MAS_PYTHON:-.venv-native-benchmark/bin/python}"
case "${1:-check}" in
 check) "$py" -m unittest experiments.tests.test_decision_review_repair experiments.tests.test_consensus_mechanism ;;
 probe) shift; "$py" -m experiments.consensus_mechanism.repair_probe "$@" ;;
 prepare|train|main) stage="$1"; shift; "$py" -m experiments.consensus_mechanism.repaired_release "$stage" "$@" ;;
 *) echo 'Usage: check | probe [--run] | prepare | train --run | main --run'; exit 2 ;;
esac
