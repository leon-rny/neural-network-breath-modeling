#!/bin/bash
# Nested selection and report (core/report.py) over results/nested/*.json of experiments/01_nested.sh: writes
# results/summary.csv, fidelity.csv, selection.csv, nested.csv (selection by accuracy) and selection_macro_f1.csv,
# nested_macro_f1.csv (selection by macro F1), and prints the paired tests to results/report*.txt.
#   experiments/02_report.sh
set -euo pipefail
cd "$(dirname "$0")/.."
python -m core.report results/nested --results results | tee results/report.txt
python -m core.report results/nested --results results --criterion macro_f1 | tee results/report_macro_f1.txt
