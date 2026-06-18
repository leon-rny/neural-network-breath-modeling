#!/bin/bash
#SBATCH --job-name=ganabl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=08:00:00
#SBATCH --output=/home/rane10/logs/ganabl.o%A_%a
#SBATCH --error=/home/rane10/logs/ganabl.e%A_%a
#SBATCH --array=0-7%50
set -euo pipefail

source /opt/miniforge/etc/profile.d/conda.sh
conda activate nnbm

# 2 lr_d x 2 loss x 2 folds = 8 runs
# single thread per task
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

REGION=mouth; INIT_SEED=0; SPLIT_SEED=42; N_FOLDS=5; EPOCHS="${EPOCHS:-500}"
LATENT_DIM=16; EMBED_DIM=8; PART_EMBED_DIM=8; ALPHA=0.05; N_COPIES=10; GAN_HIDDEN=64; LR_G=1e-3
# decimals so RUN_ID matches Python's float str (1e-4 -> 0.0001) for the ckpt-skip check
LRDS=(0.0001 0.0001 0.0002 0.0002); LOSSES=(bce hinge bce hinge); FOLDS=(1 3)
N_CFG=${#LRDS[@]}

IDX=${SLURM_ARRAY_TASK_ID:-${TASK_ID:?set SLURM_ARRAY_TASK_ID or TASK_ID=<0..7>}}
CFG=$(( IDX % N_CFG )); FOLD_IDX=$(( IDX / N_CFG ))
LRD=${LRDS[$CFG]}; LOSS=${LOSSES[$CFG]}; FOLD=${FOLDS[$FOLD_IDX]}

echo "[GANABL] task=$IDX region=$REGION fold=$FOLD lr_d=$LRD loss=$LOSS hidden=$GAN_HIDDEN"
COMMON="--model gan --region $REGION --init_seed $INIT_SEED --split_seed $SPLIT_SEED \
  --fold $FOLD --n_folds $N_FOLDS --latent_dim $LATENT_DIM --embed_dim $EMBED_DIM \
  --part_embed_dim $PART_EMBED_DIM --gan_hidden $GAN_HIDDEN --gan_loss $LOSS --gan_lr_d $LRD --lr $LR_G \
  --alpha $ALPHA --n_copies $N_COPIES"

RUN_ID="${REGION}_s${INIT_SEED}_ld${LATENT_DIM}_ed${EMBED_DIM}_gan_h${GAN_HIDDEN}_${LOSS}_lrd${LRD}_f${FOLD}_a${ALPHA}_n${N_COPIES}"
CKPT="results/gan/${RUN_ID}_checkpoint.pt"
if [ -f "$CKPT" ]; then echo "[GANABL] ckpt exists, skip train: $CKPT"; else
  PYTHONHASHSEED="$INIT_SEED" python -m core.train $COMMON --epochs "$EPOCHS"
fi

echo "==GANABL_RESULT fold=$FOLD lr_d=$LRD loss=$LOSS=="
PYTHONHASHSEED="$INIT_SEED" python -m core.tstr $COMMON --n_jobs 1 --no_summary 2>&1 \
  | grep -iE "test accuracy" | tail -1
echo "==GANABL_DONE task=$IDX=="
