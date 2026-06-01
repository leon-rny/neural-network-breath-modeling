#!/bin/bash
#SBATCH --job-name=cvae_arch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --output=/home/rane10/logs/cvae_arch.o%A_%a
#SBATCH --error=/home/rane10/logs/cvae_arch.e%A_%a
#SBATCH --array=0-1099%50
set -euo pipefail

source /opt/miniforge/etc/profile.d/conda.sh
conda activate nnbm

# 11 variants x 2 cond_part x 2 regions x 5 seeds x 5 folds = 1100 runs
VARIANTS=(conv_baseline conv_slim conv_tiny conv_large_kernel conv_asym conv_asym_no_dropout conv_baseline_dropout mlp mlp_small mlp_tiny transformer)
CONDS=(true false)
REGIONS=(mouth nose)
INIT_SEEDS=(0 1 7 42 123)
FOLDS=(1 2 3 4 5)
SPLIT_SEED=42
N_FOLDS=5
EPOCHS="${EPOCHS:-500}"

N_VARIANTS=${#VARIANTS[@]}
N_CONDS=${#CONDS[@]}
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
  echo "[ARCH] aggregate-only: building summary.csv"
  python -m ablations.cvae_ablation --mode architecture --aggregate \
    --variants "$(IFS=, ; echo "${VARIANTS[*]}")" \
    --cond_parts "$(IFS=, ; echo "${CONDS[*]}")" \
    --regions "$(IFS=, ; echo "${REGIONS[*]}")" \
    --init_seeds "$(IFS=, ; echo "${INIT_SEEDS[*]}")" \
    --folds "$(IFS=, ; echo "${FOLDS[*]}")" \
    --split_seed "$SPLIT_SEED" --n_folds "$N_FOLDS"
  exit 0
fi

IDX=${SLURM_ARRAY_TASK_ID:-${TASK_ID:?set SLURM_ARRAY_TASK_ID (via sbatch) or TASK_ID=<0..1099> for a local run}}
FOLD_IDX=$(( IDX % N_FOLDS_AX ));   IDX=$(( IDX / N_FOLDS_AX ))
SEED_IDX=$(( IDX % N_SEEDS ));      IDX=$(( IDX / N_SEEDS ))
REGION_IDX=$(( IDX % N_REGIONS ));  IDX=$(( IDX / N_REGIONS ))
COND_IDX=$(( IDX % N_CONDS ));      IDX=$(( IDX / N_CONDS ))
VARIANT_IDX=$(( IDX % N_VARIANTS ))

VARIANT=${VARIANTS[$VARIANT_IDX]}
COND=${CONDS[$COND_IDX]}
REGION=${REGIONS[$REGION_IDX]}
INIT_SEED=${INIT_SEEDS[$SEED_IDX]}
FOLD=${FOLDS[$FOLD_IDX]}

echo "[ARCH] task=${SLURM_ARRAY_TASK_ID:-$TASK_ID} variant=$VARIANT cond_part=$COND region=$REGION init_seed=$INIT_SEED fold=$FOLD"

CACHE="results/trtr/${REGION}_is${INIT_SEED}_ss${SPLIT_SEED}_fold${FOLD}of${N_FOLDS}_checkpoint.pkl"
if [ ! -f "$CACHE" ]; then
  echo "[ARCH] WARNING: TRTR cache missing ($CACHE); this task will build it (possible parallel race)."
fi

PYTHONHASHSEED="$INIT_SEED" python -m ablations.cvae_ablation --mode architecture \
  --variant "$VARIANT" \
  --cond_part "$COND" \
  --region "$REGION" \
  --init_seed "$INIT_SEED" \
  --split_seed "$SPLIT_SEED" \
  --fold "$FOLD" \
  --n_folds "$N_FOLDS" \
  --epochs "$EPOCHS" \
  --n_jobs 1 \
  --skip_existing \
  --no_summary

# Aggregate after the array finishes: AGGREGATE=1 sbatch --array=0 experiments/ablations_cvae_1_architecture.sh
