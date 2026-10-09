#!/bin/bash
#SBATCH --job-name=nested
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6G
#SBATCH --time=12:00:00
#SBATCH --output=logs/nested.o%A_%a
#SBATCH --error=logs/nested.e%A_%a
#SBATCH --array=0-639%300
#SBATCH --constraint=amd_milan|amd_rome
set -euo pipefail

# Nested LOSO (core/nested.py): one array index = one task. 2 regions x 8 outer participants x (1 outer + 7 inner tasks)
# x 5 seeds = 640 tasks; indices 0-79 are the outer tasks (evaluation, controls, TSTR, fidelity; ~2.5 h each),
# 80-639 the inner tasks (selection only; ~1.5 h each). Writes results/nested/<task>.json; finished tasks are skipped.
# Run from the repository root of a fresh run directory (a clean checkout of the commit to report) after
# experiments/00_calibration.sh; the log directory must exist before submission:
#   mkdir -p logs && sbatch experiments/01_nested.sh
#   sbatch --array=0-79 experiments/01_nested.sh      # outer tasks only
# Report: experiments/02_report.sh. The node constraint keeps all tasks on one CPU family, so reruns reproduce the
# results bit for bit; CONDA_SH / CONDA_ENV select the Python environment (requirements.txt), without conda the python
# on PATH is used.
CONDA_SH="${CONDA_SH:-/opt/miniforge/etc/profile.d/conda.sh}"
if [ -f "$CONDA_SH" ]; then source "$CONDA_SH"; conda activate "${CONDA_ENV:-nnbm}"; fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TQDM_DISABLE=1
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

REGIONS=(mouth nose)
SEEDS=(0 1 7 42 123)
N=8  # participants per region
IDX=${SLURM_ARRAY_TASK_ID:-${TASK_ID:?set SLURM_ARRAY_TASK_ID (via sbatch) or TASK_ID=<0..639> for a local run}}
PER_INNER=$(( ${#REGIONS[@]} * N * ${#SEEDS[@]} ))
INNER=$(( IDX / PER_INNER - 1 )); REST=$(( IDX % PER_INNER ))
REGION=${REGIONS[$(( REST / (N * ${#SEEDS[@]}) ))]}; REST=$(( REST % (N * ${#SEEDS[@]}) ))
OUTER=$(( REST / ${#SEEDS[@]} ))
SEED=${SEEDS[$(( REST % ${#SEEDS[@]} ))]}
if [ "$INNER" -lt 0 ]; then TASK="${REGION}_o${OUTER}_outer_s${SEED}"; else TASK="${REGION}_o${OUTER}_i${INNER}_s${SEED}"; fi
echo "[NESTED] array $IDX -> $TASK on $(hostname)"
[ -f "results/nested/${TASK}.json" ] && { echo "done already"; exit 0; }
PYTHONHASHSEED="$SEED" python -m core.nested --region "$REGION" --outer "$OUTER" --inner "$INNER" --seed "$SEED"
