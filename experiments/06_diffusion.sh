#!/bin/bash
#SBATCH --job-name=diffusion
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --output=/home/rane10/logs/diffusion.o%A_%a
#SBATCH --error=/home/rane10/logs/diffusion.e%A_%a
#SBATCH --array=0-49%50
set -euo pipefail

source /opt/miniforge/etc/profile.d/conda.sh
conda activate nnbm

# 2 regions x 5 seeds x 5 folds = 50 runs
# CV_MODE=kfold|loso (loso uses part_dropout 0.1 null token)
CV_MODE="${CV_MODE:-kfold}"
EMBED_DIM=32
PART_EMBED_DIM=8
N_STEPS="${N_STEPS:-200}"
ALPHA=0.05
N_COPIES=10
REGIONS=(mouth nose)
INIT_SEEDS=(0 1 7 42 123)
FOLDS=(1 2 3 4 5)
SPLIT_SEED=42
N_FOLDS=5
EPOCHS="${EPOCHS:-500}"

PART_DROPOUT=0.0; CV_FLAGS="--cv_mode kfold"; CVM=""; DRP=""
if [ "$CV_MODE" = "loso" ]; then PART_DROPOUT=0.1; CV_FLAGS="--cv_mode loso --part_dropout 0.1"; CVM="_loso"; DRP="_drop0.1"; fi

N_REGIONS=${#REGIONS[@]}; N_SEEDS=${#INIT_SEEDS[@]}; N_FOLDS_AX=${#FOLDS[@]}
# single thread per task
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

if [ "${AGGREGATE:-0}" = "1" ]; then
  echo "[DIFFUSION] aggregate-only ($CV_MODE)"
  python -m core.tstr --aggregate --model diffusion $CV_FLAGS \
    --regions "$(IFS=, ; echo "${REGIONS[*]}")" \
    --init_seeds "$(IFS=, ; echo "${INIT_SEEDS[*]}")" \
    --folds "$(IFS=, ; echo "${FOLDS[*]}")" \
    --split_seed "$SPLIT_SEED" --n_folds "$N_FOLDS" \
    --embed_dim "$EMBED_DIM" --alpha "$ALPHA" --n_copies "$N_COPIES"
  exit 0
fi

if [ "$CV_MODE" = "loso" ]; then
  N_DATA=$(python -c "from core.data import load_dataset, n_loso_folds; print(n_loso_folds(load_dataset('dataset')))")
  [ "$N_DATA" = "$N_FOLDS" ] || { echo "[DIFFUSION] ERROR: dataset has $N_DATA subjects but N_FOLDS=$N_FOLDS"; exit 1; }
fi

IDX=${SLURM_ARRAY_TASK_ID:-${TASK_ID:?set SLURM_ARRAY_TASK_ID or TASK_ID=<0..49>}}
FOLD_IDX=$(( IDX % N_FOLDS_AX )); IDX=$(( IDX / N_FOLDS_AX ))
SEED_IDX=$(( IDX % N_SEEDS )); IDX=$(( IDX / N_SEEDS ))
REGION_IDX=$(( IDX % N_REGIONS ))
REGION=${REGIONS[$REGION_IDX]}; INIT_SEED=${INIT_SEEDS[$SEED_IDX]}; FOLD=${FOLDS[$FOLD_IDX]}

echo "[DIFFUSION] task=${SLURM_ARRAY_TASK_ID:-$TASK_ID} cv=$CV_MODE region=$REGION seed=$INIT_SEED fold=$FOLD"

RUN_ID="${REGION}_s${INIT_SEED}_ed${EMBED_DIM}_diff_f${FOLD}${CVM}${DRP}_a${ALPHA}_n${N_COPIES}"
CKPT="results/diffusion/${RUN_ID}_checkpoint.pt"
if [ -f "$CKPT" ]; then
  echo "[DIFFUSION] checkpoint exists, skipping training: $CKPT"
else
  PYTHONHASHSEED="$INIT_SEED" python -m core.train --model diffusion --region "$REGION" $CV_FLAGS \
    --init_seed "$INIT_SEED" --split_seed "$SPLIT_SEED" --fold "$FOLD" --n_folds "$N_FOLDS" \
    --epochs "$EPOCHS" --embed_dim "$EMBED_DIM" --part_embed_dim "$PART_EMBED_DIM" --n_steps "$N_STEPS" \
    --alpha "$ALPHA" --n_copies "$N_COPIES"
fi

PYTHONHASHSEED="$INIT_SEED" python -m core.tstr --model diffusion --region "$REGION" $CV_FLAGS \
  --init_seed "$INIT_SEED" --split_seed "$SPLIT_SEED" --fold "$FOLD" --n_folds "$N_FOLDS" \
  --embed_dim "$EMBED_DIM" --alpha "$ALPHA" --n_copies "$N_COPIES" --n_jobs 1 --eval_val --no_summary

# Aggregate after the array finishes: AGGREGATE=1 [CV_MODE=loso] sbatch --array=0 experiments/06_diffusion.sh
