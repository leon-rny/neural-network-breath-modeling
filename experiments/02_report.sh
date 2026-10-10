#!/bin/bash
# Selection and report (core/report.py) over results/nested and results/kfold of experiments/01_experiment.sh, once with
# selection by accuracy and once by macro F1: results/*.csv and results/report*.txt.
#   experiments/02_report.sh
set -euo pipefail
cd "$(dirname "$0")/.."
python -m core.report results/nested --results results --kfold results/kfold | tee results/report.txt
python -m core.report results/nested --results results --kfold results/kfold --criterion macro_f1 | tee results/report_macro_f1.txt
