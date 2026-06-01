#!/bin/bash
#SBATCH --job-name=cvae_confirm_fb_lag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --output=/home/rane10/logs/cvae_confirm_fb_lag.o%A_%a
#SBATCH --error=/home/rane10/logs/cvae_confirm_fb_lag.e%A_%a
#SBATCH --array=0-299%50
set -euo pipefail

source /opt/miniforge/etc/profile.d/conda.sh
conda activate nnbm

# 6 configs x 2 regions x 5 seeds x 5 folds = 300 runs
VARIANT=conv_baseline
BETA_MAX=0.01
CONFIGS=(fb0_off fb0.1_off fb0.5_off fb0_lag5_250 fb0.1_lag5_250 fb0.5_lag5_250)
REGIONS=(mouth nose)
INIT_SEEDS=(0 1 7 42 123)
FOLDS=(1 2 3 4 5)
SPLIT_SEED=42
N_FOLDS=5
EPOCHS="${EPOCHS:-500}"

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
  echo "[DYNAMICS] aggregate-only: building summary.csv"
  python -m ablations.cvae_ablation --mode dynamics --aggregate \
    --variant "$VARIANT" \
    --beta_max "$BETA_MAX" \
    --configs "$(IFS=, ; echo "${CONFIGS[*]}")" \
    --regions "$(IFS=, ; echo "${REGIONS[*]}")" \
    --init_seeds "$(IFS=, ; echo "${INIT_SEEDS[*]}")" \
    --folds "$(IFS=, ; echo "${FOLDS[*]}")" \
    --split_seed "$SPLIT_SEED" --n_folds "$N_FOLDS"
  exit 0
fi

IDX=${SLURM_ARRAY_TASK_ID:-${TASK_ID:?set SLURM_ARRAY_TASK_ID (via sbatch) or TASK_ID=<0..299> for a local run}}
FOLD_IDX=$(( IDX % N_FOLDS_AX ));   IDX=$(( IDX / N_FOLDS_AX ))
SEED_IDX=$(( IDX % N_SEEDS ));      IDX=$(( IDX / N_SEEDS ))
REGION_IDX=$(( IDX % N_REGIONS ));  IDX=$(( IDX / N_REGIONS ))
CONFIG_IDX=$(( IDX % N_CONFIGS ))

CONFIG=${CONFIGS[$CONFIG_IDX]}
REGION=${REGIONS[$REGION_IDX]}
INIT_SEED=${INIT_SEEDS[$SEED_IDX]}
FOLD=${FOLDS[$FOLD_IDX]}

echo "[DYNAMICS] task=${SLURM_ARRAY_TASK_ID:-$TASK_ID} variant=$VARIANT beta_max=$BETA_MAX config=$CONFIG region=$REGION init_seed=$INIT_SEED fold=$FOLD"

CACHE="results/trtr/${REGION}_is${INIT_SEED}_ss${SPLIT_SEED}_fold${FOLD}of${N_FOLDS}_checkpoint.pkl"
if [ ! -f "$CACHE" ]; then
  echo "[DYNAMICS] WARNING: TRTR cache missing ($CACHE); this task will build it (possible parallel race)."
fi

PYTHONHASHSEED="$INIT_SEED" python -m ablations.cvae_ablation --mode dynamics \
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

# Aggregate after the array finishes: AGGREGATE=1 sbatch --array=0 experiments/ablations_cvae_4_confirm_freebits_lag.sh
