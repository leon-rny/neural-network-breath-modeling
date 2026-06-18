#!/bin/bash
#SBATCH --job-name=cvae_dyn_beta
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --output=/home/rane10/logs/cvae_dyn_beta.o%A_%a
#SBATCH --error=/home/rane10/logs/cvae_dyn_beta.e%A_%a
#SBATCH --array=0-349%50
set -euo pipefail

source /opt/miniforge/etc/profile.d/conda.sh
conda activate nnbm

# 7 configs x 2 regions x 5 seeds x 5 folds = 350 runs
VARIANT=conv_baseline
CONFIGS=(beta_cap_0.001 beta_cap_0.01 beta_cap_0.03 beta_cap_0.05 beta_cap_0.1 beta_cap_0.5 beta_cap_1.0)
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
  python -m ablations.cvae --mode dynamics --aggregate \
    --variant "$VARIANT" \
    --configs "$(IFS=, ; echo "${CONFIGS[*]}")" \
    --regions "$(IFS=, ; echo "${REGIONS[*]}")" \
    --init_seeds "$(IFS=, ; echo "${INIT_SEEDS[*]}")" \
    --folds "$(IFS=, ; echo "${FOLDS[*]}")" \
    --split_seed "$SPLIT_SEED" --n_folds "$N_FOLDS"
  exit 0
fi

IDX=${SLURM_ARRAY_TASK_ID:-${TASK_ID:?set SLURM_ARRAY_TASK_ID (via sbatch) or TASK_ID=<0..349> for a local run}}
FOLD_IDX=$(( IDX % N_FOLDS_AX ));   IDX=$(( IDX / N_FOLDS_AX ))
SEED_IDX=$(( IDX % N_SEEDS ));      IDX=$(( IDX / N_SEEDS ))
REGION_IDX=$(( IDX % N_REGIONS ));  IDX=$(( IDX / N_REGIONS ))
CONFIG_IDX=$(( IDX % N_CONFIGS ))

CONFIG=${CONFIGS[$CONFIG_IDX]}
REGION=${REGIONS[$REGION_IDX]}
INIT_SEED=${INIT_SEEDS[$SEED_IDX]}
FOLD=${FOLDS[$FOLD_IDX]}

echo "[DYNAMICS] task=${SLURM_ARRAY_TASK_ID:-$TASK_ID} variant=$VARIANT config=$CONFIG region=$REGION init_seed=$INIT_SEED fold=$FOLD"

CACHE="results/trtr/${REGION}_is${INIT_SEED}_ss${SPLIT_SEED}_fold${FOLD}of${N_FOLDS}_checkpoint.pkl"
if [ ! -f "$CACHE" ]; then
  echo "[DYNAMICS] WARNING: TRTR cache missing ($CACHE); this task will build it (possible parallel race)."
fi

PYTHONHASHSEED="$INIT_SEED" python -m ablations.cvae --mode dynamics \
  --variant "$VARIANT" \
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

# Aggregate after the array finishes: AGGREGATE=1 sbatch --array=0 experiments/ablations_cvae_2_dynamics_beta.sh
