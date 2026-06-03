#!/bin/bash
#SBATCH --job-name=pinn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --output=/home/rane10/logs/pinn.o%A_%a
#SBATCH --error=/home/rane10/logs/pinn.e%A_%a
#SBATCH --array=0-299%50
set -euo pipefail

source /opt/miniforge/etc/profile.d/conda.sh
conda activate nnbm

# 6 lambda_phys x 2 regions x 5 seeds x 5 folds = 300 runs
LATENT_DIM=16
EMBED_DIM=8
PART_EMBED_DIM=8
BETA_MAX=0.01
ALPHA=0.05
N_COPIES=10
LAMBDA_PHYS=(0.0 0.001 0.005 0.01 0.05 0.1)
REGIONS=(mouth nose)
INIT_SEEDS=(0 1 7 42 123)
FOLDS=(1 2 3 4 5)
SPLIT_SEED=42
N_FOLDS=5
EPOCHS="${EPOCHS:-500}"

N_LP=${#LAMBDA_PHYS[@]}
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
  echo "[PINN] aggregate-only: merging per-combo TSTR results into results/summary.csv"
  for lp in "${LAMBDA_PHYS[@]}"; do
    python -m core.tstr --aggregate --model pinn \
      --regions "$(IFS=, ; echo "${REGIONS[*]}")" \
      --init_seeds "$(IFS=, ; echo "${INIT_SEEDS[*]}")" \
      --folds "$(IFS=, ; echo "${FOLDS[*]}")" \
      --split_seed "$SPLIT_SEED" --n_folds "$N_FOLDS" \
      --latent_dim "$LATENT_DIM" --embed_dim "$EMBED_DIM" \
      --lambda_phys "$lp" --alpha "$ALPHA" --n_copies "$N_COPIES"
  done
  exit 0
fi

IDX=${SLURM_ARRAY_TASK_ID:-${TASK_ID:?set SLURM_ARRAY_TASK_ID (via sbatch) or TASK_ID=<0..299> for a local run}}
FOLD_IDX=$(( IDX % N_FOLDS_AX ));   IDX=$(( IDX / N_FOLDS_AX ))
SEED_IDX=$(( IDX % N_SEEDS ));      IDX=$(( IDX / N_SEEDS ))
REGION_IDX=$(( IDX % N_REGIONS ));  IDX=$(( IDX / N_REGIONS ))
LP_IDX=$(( IDX % N_LP ))

LAMBDA_PHYS_VAL=${LAMBDA_PHYS[$LP_IDX]}
REGION=${REGIONS[$REGION_IDX]}
INIT_SEED=${INIT_SEEDS[$SEED_IDX]}
FOLD=${FOLDS[$FOLD_IDX]}

echo "[PINN] task=${SLURM_ARRAY_TASK_ID:-$TASK_ID} region=$REGION lambda_phys=$LAMBDA_PHYS_VAL init_seed=$INIT_SEED fold=$FOLD"

PARAMS="results/pinn/params_${REGION}.npy"
if [ ! -f "$PARAMS" ]; then
  echo "[PINN] ERROR: missing CIR params ($PARAMS) — fit them before training the PINN." >&2
  exit 1
fi
CACHE="results/trtr/${REGION}_is${INIT_SEED}_ss${SPLIT_SEED}_fold${FOLD}of${N_FOLDS}_checkpoint.pkl"
if [ ! -f "$CACHE" ]; then
  echo "[PINN] WARNING: TRTR cache missing ($CACHE); TSTR will build it (possible parallel race)."
fi

RUN_ID="${REGION}_s${INIT_SEED}_ld${LATENT_DIM}_ed${EMBED_DIM}_phys${LAMBDA_PHYS_VAL}_f${FOLD}_a${ALPHA}_n${N_COPIES}"
CKPT="results/pinn/${RUN_ID}_checkpoint.pt"
if [ -f "$CKPT" ]; then
  echo "[PINN] checkpoint exists, skipping training: $CKPT"
else
  PYTHONHASHSEED="$INIT_SEED" python train_pinn.py \
    --region "$REGION" \
    --init_seed "$INIT_SEED" --split_seed "$SPLIT_SEED" \
    --fold "$FOLD" --n_folds "$N_FOLDS" \
    --lambda_phys "$LAMBDA_PHYS_VAL" \
    --latent_dim "$LATENT_DIM" --embed_dim "$EMBED_DIM" --part_embed_dim "$PART_EMBED_DIM" \
    --num_epochs "$EPOCHS" \
    --beta_max "$BETA_MAX" --alpha "$ALPHA" --n_copies "$N_COPIES"
fi

PYTHONHASHSEED="$INIT_SEED" python -m core.tstr \
  --model pinn --region "$REGION" \
  --init_seed "$INIT_SEED" --split_seed "$SPLIT_SEED" \
  --fold "$FOLD" --n_folds "$N_FOLDS" \
  --lambda_phys "$LAMBDA_PHYS_VAL" \
  --latent_dim "$LATENT_DIM" --embed_dim "$EMBED_DIM" --alpha "$ALPHA" --n_copies "$N_COPIES" \
  --n_jobs 1 --eval_val --no_summary

# Aggregate after the array finishes: AGGREGATE=1 sbatch --array=0 experiments/06_pinn.sh
