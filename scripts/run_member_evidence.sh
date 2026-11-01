#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
py="${MAS_PYTHON:-.venv-native-benchmark/bin/python}"
mode="${1:---prepare}"
case "$mode" in
 --prepare) exec "$py" -m experiments.member_evidence_collect prepare ;;
 --run)
   "$py" -m experiments.member_evidence_collect collect --run
   "$py" -m experiments.member_evidence_sequence
   "$py" -m experiments.member_evidence_run fit
   "$py" -m experiments.member_evidence_run evaluate --run ;;
 --continue)
   "$py" -m experiments.member_evidence_sequence
   "$py" -m experiments.member_evidence_run fit
   "$py" -m experiments.member_evidence_run evaluate --run ;;
 --report) exec "$py" -m experiments.member_evidence_run report ;;
 *) echo 'Usage: bash scripts/run_member_evidence.sh [--prepare|--run|--continue|--report]'; exit 2 ;;
esac
