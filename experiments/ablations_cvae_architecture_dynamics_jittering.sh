#!/bin/bash
#SBATCH --job-name=cvae_jitter
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --output=/home/rane10/logs/cvae_jitter.o%A_%a
#SBATCH --error=/home/rane10/logs/cvae_jitter.e%A_%a
#SBATCH --array=0-1799%50
set -euo pipefail

source /opt/miniforge/etc/profile.d/conda.sh
conda activate nnbm

# Expanded joint grid: architecture x beta_max x jitter x region.
# beta_max is now an explicit axis to resolve whether nose prefers a lower
# beta_max when jittering is active (the open question from the inconsistent
# pipeline, where nose jitter ran at beta=0.01 and scored ~70%).
#
#   2 variants x 3 beta_max x 6 jitter x 2 regions x 5 seeds x 5 folds = 1800 runs
VARIANTS=(conv_baseline mlp)
BETA_MAXES=(0.01 0.05 0.1)
CONFIGS=(baseline a0.025_n10 a0.05_n5 a0.05_n10 a0.1_n5 a0.1_n10)
REGIONS=(mouth nose)
INIT_SEEDS=(0 1 7 42 123)
FOLDS=(1 2 3 4 5)
SPLIT_SEED=42
N_FOLDS=5
EPOCHS="${EPOCHS:-500}"

N_VARIANTS=${#VARIANTS[@]}
N_BETAS=${#BETA_MAXES[@]}
N_CONFIGS=${#CONFIGS[@]}
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
  echo "[JITTER] aggregate-only: building summary.csv"
  python -m ablations.cvae_jittering --aggregate \
    --variants "$(IFS=, ; echo "${VARIANTS[*]}")" \
    --beta_maxes "$(IFS=, ; echo "${BETA_MAXES[*]}")" \
    --configs "$(IFS=, ; echo "${CONFIGS[*]}")" \
    --regions "$(IFS=, ; echo "${REGIONS[*]}")" \
    --init_seeds "$(IFS=, ; echo "${INIT_SEEDS[*]}")" \
    --folds "$(IFS=, ; echo "${FOLDS[*]}")" \
    --split_seed "$SPLIT_SEED" --n_folds "$N_FOLDS"
  exit 0
fi

# task id comes from SLURM under sbatch; for a local smoke test pass TASK_ID=<n> instead.
IDX=${SLURM_ARRAY_TASK_ID:-${TASK_ID:?set SLURM_ARRAY_TASK_ID (via sbatch) or TASK_ID=<0..1799> for a local run}}
# unravel: fold (fastest) -> seed -> region -> config -> beta -> variant (slowest)
FOLD_IDX=$(( IDX % N_FOLDS_AX ));   IDX=$(( IDX / N_FOLDS_AX ))
SEED_IDX=$(( IDX % N_SEEDS ));      IDX=$(( IDX / N_SEEDS ))
REGION_IDX=$(( IDX % N_REGIONS ));  IDX=$(( IDX / N_REGIONS ))
CONFIG_IDX=$(( IDX % N_CONFIGS ));  IDX=$(( IDX / N_CONFIGS ))
BETA_IDX=$(( IDX % N_BETAS ));      IDX=$(( IDX / N_BETAS ))
VARIANT_IDX=$(( IDX % N_VARIANTS ))

VARIANT=${VARIANTS[$VARIANT_IDX]}
BETA_MAX=${BETA_MAXES[$BETA_IDX]}
CONFIG=${CONFIGS[$CONFIG_IDX]}
REGION=${REGIONS[$REGION_IDX]}
INIT_SEED=${INIT_SEEDS[$SEED_IDX]}
FOLD=${FOLDS[$FOLD_IDX]}

echo "[JITTER] task=${SLURM_ARRAY_TASK_ID:-$TASK_ID} variant=$VARIANT beta_max=$BETA_MAX config=$CONFIG region=$REGION init_seed=$INIT_SEED fold=$FOLD"

CACHE="results/trtr/${REGION}_is${INIT_SEED}_ss${SPLIT_SEED}_fold${FOLD}of${N_FOLDS}_checkpoint.pkl"
if [ ! -f "$CACHE" ]; then
  echo "[JITTER] WARNING: TRTR cache missing ($CACHE); this task will build it (possible parallel race)."
fi

PYTHONHASHSEED="$INIT_SEED" python -m ablations.cvae_jittering \
  --variant "$VARIANT" \
  --beta_max "$BETA_MAX" \
  --config "$CONFIG" \
  --region "$REGION" \
  --init_seed "$INIT_SEED" \
  --split_seed "$SPLIT_SEED" \
  --fold "$FOLD" \
  --n_folds "$N_FOLDS" \
  --epochs "$EPOCHS" \
  --n_jobs 1 \
  --skip_existing \
  --no_summary

# Aggregate after the array finishes:
#   AGGREGATE=1 sbatch --array=0 experiments/ablations_cvae_architecture_dynamics_jittering.sh
# or run the aggregate block above directly on a login node.