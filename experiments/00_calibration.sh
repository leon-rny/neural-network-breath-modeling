#!/bin/bash
# Calibration of the PI-CVAE transport (core/fit_sir.py): fits the advection-diffusion + sensor response to the mean
# single-exhalation calibration recordings (dataset/sir/sir_<region>_trial_*.dat) per region and writes
# results/pinn/params_<region>.npy (A, D, v, t_s) with a manifest of the input files (.npy.json).
# Run once per run directory before experiments/01_nested.sh (from the repository root; a few minutes, no Slurm needed):
#   experiments/00_calibration.sh
set -euo pipefail
cd "$(dirname "$0")/.."
python -m core.fit_sir
