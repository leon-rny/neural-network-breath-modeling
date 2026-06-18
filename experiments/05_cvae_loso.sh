#!/bin/bash
#SBATCH --job-name=cvae_loso
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --output=/home/rane10/logs/cvae_loso.o%A_%a
#SBATCH --error=/home/rane10/logs/cvae_loso.e%A_%a
#SBATCH --array=0-79%200
set -euo pipefail

source /opt/miniforge/etc/profile.d/conda.sh
conda activate nnbm

# loso axis data-driven; 2 regions x 5 seeds x N subjects, default N=8 -> --array=0-$((2*5*N-1))%200

# single thread per task
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

LATENT_DIM=16; EMBED_DIM=8; PART_EMBED_DIM=8; FREE_BITS=0.0
BETA_MAX=0.01; ALPHA=0.05; N_COPIES=10; PART_DROPOUT=0.1  # null-token required for loso generation
REGIONS=(mouth nose); INIT_SEEDS=(0 1 7 42 123); SPLIT_SEED=42
# optional INCLUDE=comma,sorted participant subset -> subset loso; empty = full pool
INCLUDE="${INCLUDE:-}"
if [ -n "$INCLUDE" ]; then
  INC_ARG="--include_subjects $INCLUDE"; SUBTAG="_sub$(echo "$INCLUDE" | tr -d ',')"
  N_FOLDS=$(python -c "from core.data import load_dataset; d=load_dataset('dataset'); d=d[d['participant'].isin('$INCLUDE'.split(','))]; print(d['participant'].nunique())")
else
  INC_ARG=""; SUBTAG=""
  N_FOLDS=$(python -c "from core.data import load_dataset, n_loso_folds; print(n_loso_folds(load_dataset('dataset')))")
fi
FOLDS=($(seq 1 "$N_FOLDS"))
EPOCHS="${EPOCHS:-500}"

N_REGIONS=${#REGIONS[@]}; N_SEEDS=${#INIT_SEEDS[@]}; N_FOLDS_AX=${#FOLDS[@]}

if [ "${AGGREGATE:-0}" = "1" ]; then
  echo "[CVAE-LOSO] aggregate-only: merging per-combo LOSO TSTR results into results/summary.csv"
  python -m core.tstr --aggregate --model cvae_part --cv_mode loso \
    --regions "$(IFS=, ; echo "${REGIONS[*]}")" \
    --init_seeds "$(IFS=, ; echo "${INIT_SEEDS[*]}")" \
    --folds "$(IFS=, ; echo "${FOLDS[*]}")" \
    --split_seed "$SPLIT_SEED" --n_folds "$N_FOLDS" \
    --latent_dim "$LATENT_DIM" --embed_dim "$EMBED_DIM" --part_embed_dim "$PART_EMBED_DIM" \
    --free_bits "$FREE_BITS" --part_dropout "$PART_DROPOUT" --alpha "$ALPHA" --n_copies "$N_COPIES" $INC_ARG
  exit 0
fi

IDX=${SLURM_ARRAY_TASK_ID:-${TASK_ID:?set SLURM_ARRAY_TASK_ID (sbatch) or TASK_ID for a local run}}
FOLD_IDX=$(( IDX % N_FOLDS_AX )); IDX=$(( IDX / N_FOLDS_AX ))
SEED_IDX=$(( IDX % N_SEEDS )); IDX=$(( IDX / N_SEEDS ))
REGION_IDX=$(( IDX % N_REGIONS ))

REGION=${REGIONS[$REGION_IDX]}
INIT_SEED=${INIT_SEEDS[$SEED_IDX]}
FOLD=${FOLDS[$FOLD_IDX]}

echo "[CVAE-LOSO] task=${SLURM_ARRAY_TASK_ID:-$TASK_ID} region=$REGION init_seed=$INIT_SEED held_out_subject=$FOLD"

RUN_ID="${REGION}_s${INIT_SEED}_ld${LATENT_DIM}_ed${EMBED_DIM}_pd${PART_EMBED_DIM}_fb${FREE_BITS}_f${FOLD}_loso_drop${PART_DROPOUT}${SUBTAG}_a${ALPHA}_n${N_COPIES}"
CKPT="results/cvae_part/${RUN_ID}_checkpoint.pt"
if [ -f "$CKPT" ]; then
  echo "[CVAE-LOSO] checkpoint exists, skipping training: $CKPT"
else
  PYTHONHASHSEED="$INIT_SEED" python -m core.train \
    --model cvae_part --region "$REGION" --cv_mode loso \
    --init_seed "$INIT_SEED" --split_seed "$SPLIT_SEED" \
    --fold "$FOLD" --n_folds "$N_FOLDS" --epochs "$EPOCHS" \
    --part_dropout "$PART_DROPOUT" \
    --latent_dim "$LATENT_DIM" --embed_dim "$EMBED_DIM" --part_embed_dim "$PART_EMBED_DIM" \
    --beta_max "$BETA_MAX" --free_bits "$FREE_BITS" --alpha "$ALPHA" --n_copies "$N_COPIES" $INC_ARG
fi

PYTHONHASHSEED="$INIT_SEED" python -m core.tstr \
  --model cvae_part --region "$REGION" --cv_mode loso \
  --init_seed "$INIT_SEED" --split_seed "$SPLIT_SEED" \
  --fold "$FOLD" --n_folds "$N_FOLDS" \
  --part_dropout "$PART_DROPOUT" \
  --latent_dim "$LATENT_DIM" --embed_dim "$EMBED_DIM" --part_embed_dim "$PART_EMBED_DIM" \
  --free_bits "$FREE_BITS" --alpha "$ALPHA" --n_copies "$N_COPIES" $INC_ARG \
  --n_jobs 1 --eval_val --no_summary

# Aggregate after the array finishes: AGGREGATE=1 sbatch --array=0 experiments/05_cvae_loso.sh
