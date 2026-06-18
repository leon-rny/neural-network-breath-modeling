#!/bin/bash
#SBATCH --job-name=cvae_part_dropout
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --output=/home/rane10/logs/cvae_part_dropout.o%A_%a
#SBATCH --error=/home/rane10/logs/cvae_part_dropout.e%A_%a
#SBATCH --array=0-99%50
set -euo pipefail

source /opt/miniforge/etc/profile.d/conda.sh
conda activate nnbm

# 2 part_dropout x 2 regions x 5 seeds x 5 folds = 100 runs
VARIANT=conv_baseline
BETA_MAX=0.01
CONFIG=fb0_off
PART_DROPOUTS=(0.0 0.10)
REGIONS=(mouth nose)
INIT_SEEDS=(0 1 7 42 123)
FOLDS=(1 2 3 4 5)
SPLIT_SEED=42
N_FOLDS=5
EPOCHS="${EPOCHS:-500}"

N_PD=${#PART_DROPOUTS[@]}
N_REGIONS=${#REGIONS[@]}
N_SEEDS=${#INIT_SEEDS[@]}
N_FOLDS_AX=${#FOLDS[@]}

# single thread per task
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

if [ "${AGGREGATE:-0}" = "1" ]; then
  echo "[PARTDROP] aggregate-only: merging both part_dropout values into summary.csv"
  for PD in "${PART_DROPOUTS[@]}"; do
    python -m ablations.cvae --mode dynamics --aggregate \
      --variant "$VARIANT" \
      --beta_max "$BETA_MAX" \
      --config "$CONFIG" \
      --part_dropout "$PD" \
      --regions "$(IFS=, ; echo "${REGIONS[*]}")" \
      --init_seeds "$(IFS=, ; echo "${INIT_SEEDS[*]}")" \
      --folds "$(IFS=, ; echo "${FOLDS[*]}")" \
      --split_seed "$SPLIT_SEED" --n_folds "$N_FOLDS"
  done
  exit 0
fi

IDX=${SLURM_ARRAY_TASK_ID:-${TASK_ID:?set SLURM_ARRAY_TASK_ID (via sbatch) or TASK_ID=<0..99> for a local run}}
FOLD_IDX=$(( IDX % N_FOLDS_AX )); IDX=$(( IDX / N_FOLDS_AX ))
SEED_IDX=$(( IDX % N_SEEDS )); IDX=$(( IDX / N_SEEDS ))
REGION_IDX=$(( IDX % N_REGIONS )); IDX=$(( IDX / N_REGIONS ))
PD_IDX=$(( IDX % N_PD ))

PART_DROPOUT=${PART_DROPOUTS[$PD_IDX]}
REGION=${REGIONS[$REGION_IDX]}
INIT_SEED=${INIT_SEEDS[$SEED_IDX]}
FOLD=${FOLDS[$FOLD_IDX]}

echo "[PARTDROP] task=${SLURM_ARRAY_TASK_ID:-$TASK_ID} variant=$VARIANT config=$CONFIG part_dropout=$PART_DROPOUT region=$REGION init_seed=$INIT_SEED fold=$FOLD"

CACHE="results/trtr/${REGION}_is${INIT_SEED}_ss${SPLIT_SEED}_fold${FOLD}of${N_FOLDS}_checkpoint.pkl"
if [ ! -f "$CACHE" ]; then
  echo "[PARTDROP] WARNING: TRTR cache missing ($CACHE); this task will build it (possible parallel race)."
fi

PYTHONHASHSEED="$INIT_SEED" python -m ablations.cvae --mode dynamics \
  --variant "$VARIANT" \
  --beta_max "$BETA_MAX" \
  --config "$CONFIG" \
  --part_dropout "$PART_DROPOUT" \
  --region "$REGION" \
  --init_seed "$INIT_SEED" \
  --split_seed "$SPLIT_SEED" \
  --fold "$FOLD" \
  --n_folds "$N_FOLDS" \
  --epochs "$EPOCHS" \
  --n_jobs 1 \
  --skip_existing \
  --no_summary

# Aggregate after the array finishes: AGGREGATE=1 sbatch --array=0 experiments/15_cvae_dropout.sh
