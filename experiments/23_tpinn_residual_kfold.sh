#!/bin/bash
#SBATCH --job-name=tpinnres_kfold
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --output=/home/rane10/logs/tpinnres_kfold.o%A_%a
#SBATCH --error=/home/rane10/logs/tpinnres_kfold.e%A_%a
#SBATCH --array=0-49%50
set -euo pipefail

source /opt/miniforge/etc/profile.d/conda.sh
conda activate nnbm

# 2 regions x 5 seeds x 5 folds = 50 runs
LATENT_DIM=16
EMBED_DIM=8
PART_EMBED_DIM=8
BETA_MAX=0.01
ALPHA=0.05
N_COPIES=10
REGIONS=(mouth nose)
INIT_SEEDS=(0 1 7 42 123)
FOLDS=(1 2 3 4 5)
SPLIT_SEED=42
N_FOLDS=5
EPOCHS="${EPOCHS:-500}"

N_REGIONS=${#REGIONS[@]}
N_SEEDS=${#INIT_SEEDS[@]}
N_FOLDS_AX=${#FOLDS[@]}

# single thread per task
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

if [ "${AGGREGATE:-0}" = "1" ]; then
  echo "[TPINNRES-KFOLD] aggregate-only: merging per-combo TSTR results into results/summary.csv"
  python -m core.tstr --aggregate --model tpinn --phys_residual \
    --regions "$(IFS=, ; echo "${REGIONS[*]}")" \
    --init_seeds "$(IFS=, ; echo "${INIT_SEEDS[*]}")" \
    --folds "$(IFS=, ; echo "${FOLDS[*]}")" \
    --split_seed "$SPLIT_SEED" --n_folds "$N_FOLDS" \
    --latent_dim "$LATENT_DIM" --embed_dim "$EMBED_DIM" --alpha "$ALPHA" --n_copies "$N_COPIES"
  exit 0
fi

IDX=${SLURM_ARRAY_TASK_ID:-${TASK_ID:?set SLURM_ARRAY_TASK_ID (sbatch) or TASK_ID=<0..49>}}
FOLD_IDX=$(( IDX % N_FOLDS_AX )); IDX=$(( IDX / N_FOLDS_AX ))
SEED_IDX=$(( IDX % N_SEEDS )); IDX=$(( IDX / N_SEEDS ))
REGION_IDX=$(( IDX % N_REGIONS ))

REGION=${REGIONS[$REGION_IDX]}
INIT_SEED=${INIT_SEEDS[$SEED_IDX]}
FOLD=${FOLDS[$FOLD_IDX]}

echo "[TPINNRES-KFOLD] task=${SLURM_ARRAY_TASK_ID:-$TASK_ID} region=$REGION init_seed=$INIT_SEED fold=$FOLD"

PARAMS="results/pinn/params_${REGION}.npy"
[ -f "$PARAMS" ] || { echo "[TPINNRES-KFOLD] ERROR: missing CIR params ($PARAMS)."; exit 1; }

RUN_ID="${REGION}_s${INIT_SEED}_ld${LATENT_DIM}_ed${EMBED_DIM}_tphys_res_f${FOLD}_a${ALPHA}_n${N_COPIES}"
CKPT="results/tpinn/${RUN_ID}_checkpoint.pt"
if [ -f "$CKPT" ]; then
  echo "[TPINNRES-KFOLD] checkpoint exists, skipping training: $CKPT"
else
  PYTHONHASHSEED="$INIT_SEED" python -m core.train \
    --model tpinn --region "$REGION" --phys_residual \
    --init_seed "$INIT_SEED" --split_seed "$SPLIT_SEED" \
    --fold "$FOLD" --n_folds "$N_FOLDS" --epochs "$EPOCHS" \
    --latent_dim "$LATENT_DIM" --embed_dim "$EMBED_DIM" --part_embed_dim "$PART_EMBED_DIM" \
    --beta_max "$BETA_MAX" --alpha "$ALPHA" --n_copies "$N_COPIES"
fi

PYTHONHASHSEED="$INIT_SEED" python -m core.tstr \
  --model tpinn --region "$REGION" --phys_residual \
  --init_seed "$INIT_SEED" --split_seed "$SPLIT_SEED" \
  --fold "$FOLD" --n_folds "$N_FOLDS" \
  --latent_dim "$LATENT_DIM" --embed_dim "$EMBED_DIM" --alpha "$ALPHA" --n_copies "$N_COPIES" \
  --n_jobs 1 --eval_val --no_summary

# Aggregate after the array finishes: AGGREGATE=1 sbatch --array=0 experiments/23_tpinn_residual_kfold.sh
