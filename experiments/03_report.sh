#!/bin/bash
# Nested selection and report (core/report.py) over results/nested/*.json of experiments/01_nested.sh and the k-fold
# fidelity results/kfold/*.json of experiments/02_kfold.sh: writes results/summary.csv, fidelity.csv,
# fidelity_kfold.csv, selection.csv, nested.csv (selection by accuracy) and selection_macro_f1.csv, nested_macro_f1.csv
# (selection by macro F1), and prints the paired tests to results/report*.txt.
#   experiments/03_report.sh
set -euo pipefail
cd "$(dirname "$0")/.."
python -m core.report results/nested --results results --kfold results/kfold | tee results/report.txt
python -m core.report results/nested --results results --kfold results/kfold --criterion macro_f1 | tee results/report_macro_f1.txt
