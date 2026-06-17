#!/bin/bash
#SBATCH --job-name=gan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --output=/home/rane10/logs/gan.o%A_%a
#SBATCH --error=/home/rane10/logs/gan.e%A_%a
#SBATCH --array=0-49%50
set -euo pipefail

source /opt/miniforge/etc/profile.d/conda.sh
conda activate nnbm

# Conditional 1D GAN generator (completes the CVAE/VAE/GAN benchmark). CV_MODE=kfold|loso
# (loso: part_dropout 0.1 null token in G). Compare to CVAE (.798/.664 kfold, .666/.483 loso).
CV_MODE="${CV_MODE:-kfold}"
LATENT_DIM=16; EMBED_DIM=8; PART_EMBED_DIM=8; ALPHA=0.05; N_COPIES=10
REGIONS=(mouth nose); INIT_SEEDS=(0 1 7 42 123); FOLDS=(1 2 3 4 5); SPLIT_SEED=42; N_FOLDS=5
EPOCHS="${EPOCHS:-500}"

CV_FLAGS="--cv_mode kfold"; CVM=""; DRP=""
if [ "$CV_MODE" = "loso" ]; then CV_FLAGS="--cv_mode loso --part_dropout 0.1"; CVM="_loso"; DRP="_drop0.1"; fi
N_REGIONS=${#REGIONS[@]}; N_SEEDS=${#INIT_SEEDS[@]}; N_FOLDS_AX=${#FOLDS[@]}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

if [ "${AGGREGATE:-0}" = "1" ]; then
  echo "[GAN] aggregate-only ($CV_MODE)"
  python -m core.tstr --aggregate --model gan $CV_FLAGS \
    --regions "$(IFS=, ; echo "${REGIONS[*]}")" --init_seeds "$(IFS=, ; echo "${INIT_SEEDS[*]}")" \
    --folds "$(IFS=, ; echo "${FOLDS[*]}")" --split_seed "$SPLIT_SEED" --n_folds "$N_FOLDS" \
    --latent_dim "$LATENT_DIM" --embed_dim "$EMBED_DIM" --alpha "$ALPHA" --n_copies "$N_COPIES"
  exit 0
fi

if [ "$CV_MODE" = "loso" ]; then
  N_DATA=$(python -c "from core.data import load_dataset, n_loso_folds; print(n_loso_folds(load_dataset('dataset')))")
  [ "$N_DATA" = "$N_FOLDS" ] || { echo "[GAN] ERROR: $N_DATA subjects != N_FOLDS=$N_FOLDS"; exit 1; }
fi

IDX=${SLURM_ARRAY_TASK_ID:-${TASK_ID:?set SLURM_ARRAY_TASK_ID or TASK_ID=<0..49>}}
FOLD_IDX=$(( IDX % N_FOLDS_AX )); IDX=$(( IDX / N_FOLDS_AX ))
SEED_IDX=$(( IDX % N_SEEDS ));    IDX=$(( IDX / N_SEEDS ))
REGION_IDX=$(( IDX % N_REGIONS ))
REGION=${REGIONS[$REGION_IDX]}; INIT_SEED=${INIT_SEEDS[$SEED_IDX]}; FOLD=${FOLDS[$FOLD_IDX]}

echo "[GAN] task=${SLURM_ARRAY_TASK_ID:-$TASK_ID} cv=$CV_MODE region=$REGION seed=$INIT_SEED fold=$FOLD"

RUN_ID="${REGION}_s${INIT_SEED}_ld${LATENT_DIM}_ed${EMBED_DIM}_gan_f${FOLD}${CVM}${DRP}_a${ALPHA}_n${N_COPIES}"
CKPT="results/gan/${RUN_ID}_checkpoint.pt"
if [ -f "$CKPT" ]; then
  echo "[GAN] checkpoint exists, skipping training: $CKPT"
else
  PYTHONHASHSEED="$INIT_SEED" python -m core.train --model gan --region "$REGION" $CV_FLAGS \
    --init_seed "$INIT_SEED" --split_seed "$SPLIT_SEED" --fold "$FOLD" --n_folds "$N_FOLDS" --epochs "$EPOCHS" \
    --latent_dim "$LATENT_DIM" --embed_dim "$EMBED_DIM" --part_embed_dim "$PART_EMBED_DIM" \
    --alpha "$ALPHA" --n_copies "$N_COPIES"
fi

PYTHONHASHSEED="$INIT_SEED" python -m core.tstr --model gan --region "$REGION" $CV_FLAGS \
  --init_seed "$INIT_SEED" --split_seed "$SPLIT_SEED" --fold "$FOLD" --n_folds "$N_FOLDS" \
  --latent_dim "$LATENT_DIM" --embed_dim "$EMBED_DIM" --alpha "$ALPHA" --n_copies "$N_COPIES" \
  --n_jobs 1 --eval_val --no_summary

# Aggregate: AGGREGATE=1 [CV_MODE=loso] sbatch --array=0 experiments/23_gan.sh
