#!/bin/bash
#SBATCH --job-name=subset_search
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=08:00:00
#SBATCH --output=/home/rane10/logs/subset.o%A_%a
#SBATCH --error=/home/rane10/logs/subset.e%A_%a
#SBATCH --array=0-929%200
set -euo pipefail

source /opt/miniforge/etc/profile.d/conda.sh
conda activate nnbm
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# Phase 2 — exhaustive participant-subset search via the cheap TRTR proxy (no generator).
# All subsets of size >= MIN_SIZE of the 8 participants (=93) x 2 regions x 5 folds = 930 -> --array=0-929.
# seed 0 only for the search (cheap); finalists get full-seed + LOSO validation separately.
# Aggregate after:  AGGREGATE=1 sbatch --array=0 experiments/28_subset_search.sh   (or python -m core.subset_search aggregate)
MIN_SIZE=5
REGIONS=(mouth nose)
FOLDS=(1 2 3 4 5)
INIT_SEED=0
SPLIT_SEED=42
N_FOLDS=5

N_REGIONS=${#REGIONS[@]}; N_FOLDS_AX=${#FOLDS[@]}
N_SUBSETS=$(python -m core.subset_search enumerate --min_size "$MIN_SIZE" | head -1)

if [ "${AGGREGATE:-0}" = "1" ]; then
  echo "[SUBSET] aggregate-only -> results/subset_search.csv"
  python -m core.subset_search aggregate --min_size "$MIN_SIZE" --init_seed "$INIT_SEED" \
    --regions "$(IFS=, ; echo "${REGIONS[*]}")" --folds "$(IFS=, ; echo "${FOLDS[*]}")"
  exit 0
fi

IDX=${SLURM_ARRAY_TASK_ID:-${TASK_ID:?set SLURM_ARRAY_TASK_ID (sbatch) or TASK_ID for a local run}}
FOLD_IDX=$(( IDX % N_FOLDS_AX )); IDX=$(( IDX / N_FOLDS_AX ))
REGION_IDX=$(( IDX % N_REGIONS )); IDX=$(( IDX / N_REGIONS ))
SUBSET_IDX=$IDX
if [ "$SUBSET_IDX" -ge "$N_SUBSETS" ]; then echo "[SUBSET] idx $SUBSET_IDX >= $N_SUBSETS, nothing to do"; exit 0; fi

REGION=${REGIONS[$REGION_IDX]}
FOLD=${FOLDS[$FOLD_IDX]}
SUBSET=$(python -m core.subset_search enumerate --min_size "$MIN_SIZE" --index "$SUBSET_IDX")

echo "[SUBSET] task=$IDX region=$REGION fold=$FOLD subset=$SUBSET"
PYTHONHASHSEED="$INIT_SEED" python -m core.tstr --model trtr --region "$REGION" \
  --include_subjects "$SUBSET" --cv_mode kfold \
  --init_seed "$INIT_SEED" --split_seed "$SPLIT_SEED" --fold "$FOLD" --n_folds "$N_FOLDS" \
  --n_jobs 1 --no_summary
