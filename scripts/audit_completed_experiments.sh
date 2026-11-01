#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${MAS_PYTHON:-.venv-native-benchmark/bin/python}"
"$PYTHON" -m experiments.audits.final_integrity_audit
"$PYTHON" -m experiments.audits.selection_value_audit
"$PYTHON" -m experiments.audits.resource_fairness_audit
"$PYTHON" -m experiments.audits.method_identity_audit
"$PYTHON" -m experiments.audits.consolidate_experiments
"$PYTHON" -m unittest experiments.tests.test_consolidation_method_identity experiments.tests.test_selection_value_audit -q
