#!/bin/bash
# Calibration of the PI-CVAE transport (core/calibrate.py) per region -> results/pinn/params_<region>.npy (+ manifest).
# Once per run directory before experiments/01_experiment.sh, from the repository root (a few minutes, no Slurm):
#   experiments/00_calibration.sh
set -euo pipefail
cd "$(dirname "$0")/.."
python -m core.calibrate
