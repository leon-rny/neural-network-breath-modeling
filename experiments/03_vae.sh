#!/bin/bash
#SBATCH --job-name=vae
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --output=/home/rane10/logs/vae.o%A_%a
#SBATCH --error=/home/rane10/logs/vae.e%A_%a
#SBATCH --array=0-149%50
set -euo pipefail

source /opt/miniforge/etc/profile.d/conda.sh
conda activate nnbm

# 3 free_bits x 2 regions x 5 seeds x 5 folds = 150 runs
MODEL=vae
LATENT_DIM=16
FREE_BITS=(0.0 0.1 2.0)
REGIONS=(mouth nose)
INIT_SEEDS=(0 1 7 42 123)
FOLDS=(1 2 3 4 5)
SPLIT_SEED=42
N_FOLDS=5
EPOCHS="${EPOCHS:-500}"

N_FB=${#FREE_BITS[@]}
N_REGIONS=${#REGIONS[@]}
N_SEEDS=${#INIT_SEEDS[@]}
N_FOLDS_AX=${#FOLDS[@]}

# single thread per task
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

if [ "${AGGREGATE:-0}" = "1" ]; then
  echo "[VAE] aggregate-only: merging per-combo TSTR results into results/summary.csv"
  for fb in "${FREE_BITS[@]}"; do
    python -m core.tstr --aggregate --model "$MODEL" \
      --regions "$(IFS=, ; echo "${REGIONS[*]}")" \
      --init_seeds "$(IFS=, ; echo "${INIT_SEEDS[*]}")" \
      --folds "$(IFS=, ; echo "${FOLDS[*]}")" \
      --split_seed "$SPLIT_SEED" --n_folds "$N_FOLDS" \
      --latent_dim "$LATENT_DIM" --free_bits "$fb"
  done
  exit 0
fi

IDX=${SLURM_ARRAY_TASK_ID:-${TASK_ID:?set SLURM_ARRAY_TASK_ID (via sbatch) or TASK_ID=<0..149> for a local run}}
FOLD_IDX=$(( IDX % N_FOLDS_AX )); IDX=$(( IDX / N_FOLDS_AX ))
SEED_IDX=$(( IDX % N_SEEDS )); IDX=$(( IDX / N_SEEDS ))
REGION_IDX=$(( IDX % N_REGIONS )); IDX=$(( IDX / N_REGIONS ))
FB_IDX=$(( IDX % N_FB ))

FREE_BITS_VAL=${FREE_BITS[$FB_IDX]}
REGION=${REGIONS[$REGION_IDX]}
INIT_SEED=${INIT_SEEDS[$SEED_IDX]}
FOLD=${FOLDS[$FOLD_IDX]}

echo "[VAE] task=${SLURM_ARRAY_TASK_ID:-$TASK_ID} model=$MODEL region=$REGION free_bits=$FREE_BITS_VAL init_seed=$INIT_SEED fold=$FOLD"

CACHE="results/trtr/${REGION}_is${INIT_SEED}_ss${SPLIT_SEED}_fold${FOLD}of${N_FOLDS}_checkpoint.pkl"
if [ ! -f "$CACHE" ]; then
  echo "[VAE] WARNING: TRTR cache missing ($CACHE); TSTR will build it (possible parallel race)."
fi

RUN_ID="${REGION}_s${INIT_SEED}_ld${LATENT_DIM}_fb${FREE_BITS_VAL}_f${FOLD}"
CKPT="results/${MODEL}/${RUN_ID}_checkpoint.pt"
if [ -f "$CKPT" ]; then
  echo "[VAE] checkpoint exists, skipping training: $CKPT"
else
  PYTHONHASHSEED="$INIT_SEED" python -m core.train \
    --model "$MODEL" --region "$REGION" \
    --init_seed "$INIT_SEED" --split_seed "$SPLIT_SEED" \
    --fold "$FOLD" --n_folds "$N_FOLDS" \
    --epochs "$EPOCHS" \
    --latent_dim "$LATENT_DIM" --free_bits "$FREE_BITS_VAL"
fi

PYTHONHASHSEED="$INIT_SEED" python -m core.tstr \
  --model "$MODEL" --region "$REGION" \
  --init_seed "$INIT_SEED" --split_seed "$SPLIT_SEED" \
  --fold "$FOLD" --n_folds "$N_FOLDS" \
  --latent_dim "$LATENT_DIM" --free_bits "$FREE_BITS_VAL" \
  --n_jobs 1 --eval_val --no_summary

# Aggregate after the array finishes: AGGREGATE=1 sbatch --array=0 experiments/vae.sh
