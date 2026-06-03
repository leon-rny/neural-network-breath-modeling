#!/bin/bash
#SBATCH --job-name=trtr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --output=/home/rane10/logs/trtr.o%A_%a
#SBATCH --error=/home/rane10/logs/trtr.e%A_%a
#SBATCH --array=0-49%50
set -euo pipefail

source /opt/miniforge/etc/profile.d/conda.sh
conda activate nnbm

# 2 regions x 5 seeds x 5 folds = 50 runs
MODEL=trtr
REGIONS=(mouth nose)
INIT_SEEDS=(0 1 7 42 123)
FOLDS=(1 2 3 4 5)
SPLIT_SEED=42
N_FOLDS=5
FORCE_REBUILD="${FORCE_REBUILD:-0}"

N_REGIONS=${#REGIONS[@]}
N_SEEDS=${#INIT_SEEDS[@]}
N_FOLDS_AX=${#FOLDS[@]}

# single thread per task
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [ "${AGGREGATE:-0}" = "1" ]; then
  echo "[TRTR] aggregate-only: merging per-combo results into results/summary.csv"
  python -m core.tstr --aggregate --model "$MODEL" \
    --regions "$(IFS=, ; echo "${REGIONS[*]}")" \
    --init_seeds "$(IFS=, ; echo "${INIT_SEEDS[*]}")" \
    --folds "$(IFS=, ; echo "${FOLDS[*]}")" \
    --split_seed "$SPLIT_SEED" --n_folds "$N_FOLDS"
  exit 0
fi

IDX=${SLURM_ARRAY_TASK_ID:-${TASK_ID:?set SLURM_ARRAY_TASK_ID (via sbatch) or TASK_ID=<0..49> for a local run}}
FOLD_IDX=$(( IDX % N_FOLDS_AX ));   IDX=$(( IDX / N_FOLDS_AX ))
SEED_IDX=$(( IDX % N_SEEDS ));      IDX=$(( IDX / N_SEEDS ))
REGION_IDX=$(( IDX % N_REGIONS ))

REGION=${REGIONS[$REGION_IDX]}
INIT_SEED=${INIT_SEEDS[$SEED_IDX]}
FOLD=${FOLDS[$FOLD_IDX]}

echo "[TRTR] task=${SLURM_ARRAY_TASK_ID:-$TASK_ID} region=$REGION init_seed=$INIT_SEED fold=$FOLD force_rebuild=$FORCE_REBUILD"

CACHE="results/trtr/${REGION}_is${INIT_SEED}_ss${SPLIT_SEED}_fold${FOLD}of${N_FOLDS}_checkpoint.pkl"
if [ ! -f "$CACHE" ]; then
  echo "[TRTR] note: TRTR cache missing ($CACHE); it will be built now."
fi

REBUILD=""
[ "$FORCE_REBUILD" = "1" ] && REBUILD="--force_rebuild"

PYTHONHASHSEED="$INIT_SEED" python -m core.tstr \
  --model "$MODEL" --region "$REGION" \
  --init_seed "$INIT_SEED" --split_seed "$SPLIT_SEED" \
  --fold "$FOLD" --n_folds "$N_FOLDS" \
  --n_jobs 1 --no_summary $REBUILD

# Aggregate after the array finishes: AGGREGATE=1 sbatch --array=0 experiments/trtr.sh
