#!/bin/bash
#SBATCH --job-name=diffabl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=08:00:00
#SBATCH --output=/home/rane10/logs/diffabl.o%A_%a
#SBATCH --error=/home/rane10/logs/diffabl.e%A_%a
#SBATCH --array=0-7%50
set -euo pipefail

source /opt/miniforge/etc/profile.d/conda.sh
conda activate nnbm

# 2 hidden x 2 n_steps x 2 folds = 8 runs (each evaluated at guidance 1, 3, 5)
# single thread per task
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

REGION=mouth; INIT_SEED=0; SPLIT_SEED=42; N_FOLDS=5; EPOCHS="${EPOCHS:-500}"
EMBED_DIM=8; PART_EMBED_DIM=8; ALPHA=0.05; N_COPIES=10
HIDDENS=(64 64 128 128); NSTEPS=(200 500 200 500); FOLDS=(1 3)
N_CFG=${#HIDDENS[@]}

IDX=${SLURM_ARRAY_TASK_ID:-${TASK_ID:?set SLURM_ARRAY_TASK_ID or TASK_ID=<0..7>}}
CFG=$(( IDX % N_CFG )); FOLD_IDX=$(( IDX / N_CFG ))
H=${HIDDENS[$CFG]}; ST=${NSTEPS[$CFG]}; FOLD=${FOLDS[$FOLD_IDX]}

echo "[DIFFABL] task=$IDX region=$REGION fold=$FOLD hidden=$H n_steps=$ST"
COMMON="--model diffusion --region $REGION --init_seed $INIT_SEED --split_seed $SPLIT_SEED \
  --fold $FOLD --n_folds $N_FOLDS --embed_dim $EMBED_DIM --part_embed_dim $PART_EMBED_DIM \
  --diff_hidden $H --n_steps $ST --alpha $ALPHA --n_copies $N_COPIES"

RUN_ID="${REGION}_s${INIT_SEED}_ed${EMBED_DIM}_diff_h${H}_st${ST}_f${FOLD}_a${ALPHA}_n${N_COPIES}"
CKPT="results/diffusion/${RUN_ID}_checkpoint.pt"
if [ -f "$CKPT" ]; then echo "[DIFFABL] ckpt exists, skip train: $CKPT"; else
  PYTHONHASHSEED="$INIT_SEED" python -m core.train $COMMON --epochs "$EPOCHS"
fi

for G in 1 3 5; do
  echo "==DIFFABL_RESULT fold=$FOLD hidden=$H n_steps=$ST guidance=$G=="
  PYTHONHASHSEED="$INIT_SEED" python -m core.tstr $COMMON --guidance "$G" --n_jobs 1 --no_summary 2>&1 \
    | grep -iE "test accuracy" | tail -1
done
echo "==DIFFABL_DONE task=$IDX=="
