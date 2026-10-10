#!/bin/bash
#SBATCH --job-name=kfold
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6G
#SBATCH --time=6:00:00
#SBATCH --output=logs/kfold.o%A_%a
#SBATCH --error=logs/kfold.e%A_%a
#SBATCH --array=0-49
#SBATCH --constraint=amd_milan|amd_rome
set -euo pipefail

# Within-participant (k-fold) fidelity (core/kfold.py), the counterpart of the LOSO fidelity of experiments/01_nested.sh:
# one array index = one task. 2 regions x 5 folds x 5 seeds = 50 tasks (~1.5 h each: PI-CVAE + CVAE training, no
# classifiers). Writes results/kfold/<task>.json; finished tasks are skipped. Independent of 01_nested.sh (both can run at
# the same time) after experiments/00_calibration.sh, from the repository root of the run directory:
#   mkdir -p logs && sbatch experiments/02_kfold.sh
# Report: experiments/03_report.sh. CONDA_SH / CONDA_ENV as in 01_nested.sh.
CONDA_SH="${CONDA_SH:-/opt/miniforge/etc/profile.d/conda.sh}"
if [ -f "$CONDA_SH" ]; then source "$CONDA_SH"; conda activate "${CONDA_ENV:-nnbm}"; fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TQDM_DISABLE=1
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

REGIONS=(mouth nose)
SEEDS=(0 1 7 42 123)
FOLDS=5
IDX=${SLURM_ARRAY_TASK_ID:-${TASK_ID:?set SLURM_ARRAY_TASK_ID (via sbatch) or TASK_ID=<0..49> for a local run}}
REGION=${REGIONS[$(( IDX / (FOLDS * ${#SEEDS[@]}) ))]}; REST=$(( IDX % (FOLDS * ${#SEEDS[@]}) ))
FOLD=$(( REST / ${#SEEDS[@]} ))
SEED=${SEEDS[$(( REST % ${#SEEDS[@]} ))]}
TASK="${REGION}_k${FOLD}_s${SEED}"
echo "[KFOLD] array $IDX -> $TASK on $(hostname)"
[ -f "results/kfold/${TASK}.json" ] && { echo "done already"; exit 0; }
PYTHONHASHSEED="$SEED" python -m core.kfold --region "$REGION" --fold "$FOLD" --seed "$SEED"
