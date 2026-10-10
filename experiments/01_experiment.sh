#!/bin/bash
#SBATCH --job-name=experiment
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6G
#SBATCH --time=12:00:00
#SBATCH --output=logs/experiment.o%A_%a
#SBATCH --error=logs/experiment.e%A_%a
#SBATCH --array=0-689%300
#SBATCH --constraint=amd_milan|amd_rome
set -euo pipefail

# One array index = one task of core/nested.py, finished tasks are skipped:
#   0-639    nested LOSO, 2 regions x 8 outer participants x (1 outer + 7 inner) x 5 seeds -> results/nested/<task>.json
#            (0-79 outer tasks ~0.5-1.5 h, inner tasks ~0.3-1 h)
#   640-689  k-fold fidelity, 2 regions x 5 folds x 5 seeds -> results/kfold/<task>.json (~0.5 h)
# From the root of a fresh run directory (clean checkout of the commit to report), after experiments/00_calibration.sh:
#   mkdir -p logs && sbatch experiments/01_experiment.sh
# The node constraint keeps all tasks on one CPU family, so reruns are bit-identical. CONDA_SH / CONDA_ENV select the
# environment (requirements.txt), without conda the python on PATH is used.
CONDA_SH="${CONDA_SH:-/opt/miniforge/etc/profile.d/conda.sh}"
if [ -f "$CONDA_SH" ]; then
    source "$CONDA_SH"
    conda activate "${CONDA_ENV:-nnbm}"
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TQDM_DISABLE=1
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

REGIONS=(mouth nose)
SEEDS=(0 1 7 42 123)
N=8  # participants per region
FOLDS=5
IDX=${SLURM_ARRAY_TASK_ID:-${TASK_ID:?set SLURM_ARRAY_TASK_ID (via sbatch) or TASK_ID=<0..689> for a local run}}
N_NESTED=$(( ${#REGIONS[@]} * N * N * ${#SEEDS[@]} ))
if [ "$IDX" -lt "$N_NESTED" ]; then
    # nested LOSO: inner -1 (outer task) .. 6, then region, outer participant, seed
    PER_INNER=$(( ${#REGIONS[@]} * N * ${#SEEDS[@]} ))
    INNER=$(( IDX / PER_INNER - 1 ))
    REST=$(( IDX % PER_INNER ))
    REGION=${REGIONS[$(( REST / (N * ${#SEEDS[@]}) ))]}
    REST=$(( REST % (N * ${#SEEDS[@]}) ))
    OUTER=$(( REST / ${#SEEDS[@]} ))
    SEED=${SEEDS[$(( REST % ${#SEEDS[@]} ))]}
    if [ "$INNER" -lt 0 ]; then
        TASK="${REGION}_o${OUTER}_outer_s${SEED}"
    else
        TASK="${REGION}_o${OUTER}_i${INNER}_s${SEED}"
    fi
    DIR=nested
    ARGS="--outer $OUTER --inner $INNER"
else
    # k-fold: region, fold, seed
    REST=$(( IDX - N_NESTED ))
    REGION=${REGIONS[$(( REST / (FOLDS * ${#SEEDS[@]}) ))]}
    REST=$(( REST % (FOLDS * ${#SEEDS[@]}) ))
    FOLD=$(( REST / ${#SEEDS[@]} ))
    SEED=${SEEDS[$(( REST % ${#SEEDS[@]} ))]}
    TASK="${REGION}_k${FOLD}_s${SEED}"
    DIR=kfold
    ARGS="--fold $FOLD"
fi
echo "[EXPERIMENT] array $IDX -> $TASK on $(hostname)"
if [ -f "results/${DIR}/${TASK}.json" ]; then
    echo "done already"
    exit 0
fi
PYTHONHASHSEED="$SEED" python -m core.nested --region "$REGION" --seed "$SEED" $ARGS
