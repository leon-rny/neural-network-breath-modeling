#!/bin/bash
#SBATCH --job-name=tpinnps_loso
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --output=/home/rane10/logs/tpinnps_loso.o%A_%a
#SBATCH --error=/home/rane10/logs/tpinnps_loso.e%A_%a
#SBATCH --array=0-79%200
set -euo pipefail

source /opt/miniforge/etc/profile.d/conda.sh
conda activate nnbm

# 2 regions x 5 seeds x N subjects (loso axis data-driven) = 2*5*N runs
# loso axis data-driven; size with --array=0-$((2*5*N-1))%200
LATENT_DIM=16
EMBED_DIM=8
PART_EMBED_DIM=8
BETA_MAX=0.01
ALPHA=0.05
N_COPIES=10
PART_DROPOUT=0.1
REGIONS=(mouth nose)
INIT_SEEDS=(0 1 7 42 123)
SPLIT_SEED=42
# subject count is data-driven; folds = seq 1 N_FOLDS
N_FOLDS=$(python -c "from core.data import load_dataset, n_loso_folds; print(n_loso_folds(load_dataset('dataset')))")
FOLDS=($(seq 1 "$N_FOLDS")) # 1-indexed held-out subject
EPOCHS="${EPOCHS:-500}"

N_REGIONS=${#REGIONS[@]}
N_SEEDS=${#INIT_SEEDS[@]}
N_FOLDS_AX=${#FOLDS[@]}

# single thread per task
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

if [ "${AGGREGATE:-0}" = "1" ]; then
  echo "[TPINNPS-LOSO] aggregate-only: merging per-combo LOSO TSTR results into results/summary.csv"
  python -m core.tstr --aggregate --model tpinn --cv_mode loso --parametric_source \
    --regions "$(IFS=, ; echo "${REGIONS[*]}")" \
    --init_seeds "$(IFS=, ; echo "${INIT_SEEDS[*]}")" \
    --folds "$(IFS=, ; echo "${FOLDS[*]}")" \
    --split_seed "$SPLIT_SEED" --n_folds "$N_FOLDS" \
    --latent_dim "$LATENT_DIM" --embed_dim "$EMBED_DIM" --part_dropout "$PART_DROPOUT" --alpha "$ALPHA" --n_copies "$N_COPIES"
  exit 0
fi

IDX=${SLURM_ARRAY_TASK_ID:-${TASK_ID:?set SLURM_ARRAY_TASK_ID (sbatch) or TASK_ID for a local run}}
FOLD_IDX=$(( IDX % N_FOLDS_AX )); IDX=$(( IDX / N_FOLDS_AX ))
SEED_IDX=$(( IDX % N_SEEDS )); IDX=$(( IDX / N_SEEDS ))
REGION_IDX=$(( IDX % N_REGIONS ))

REGION=${REGIONS[$REGION_IDX]}
INIT_SEED=${INIT_SEEDS[$SEED_IDX]}
FOLD=${FOLDS[$FOLD_IDX]}

echo "[TPINNPS-LOSO] task=${SLURM_ARRAY_TASK_ID:-$TASK_ID} region=$REGION init_seed=$INIT_SEED held_out_subject=$FOLD"

PARAMS="results/pinn/params_${REGION}.npy"
[ -f "$PARAMS" ] || { echo "[TPINNPS-LOSO] ERROR: missing CIR params ($PARAMS)."; exit 1; }

RUN_ID="${REGION}_s${INIT_SEED}_ld${LATENT_DIM}_ed${EMBED_DIM}_tphys_ps_f${FOLD}_loso_drop${PART_DROPOUT}_a${ALPHA}_n${N_COPIES}"
CKPT="results/tpinn/${RUN_ID}_checkpoint.pt"
if [ -f "$CKPT" ]; then
  echo "[TPINNPS-LOSO] checkpoint exists, skipping training: $CKPT"
else
  PYTHONHASHSEED="$INIT_SEED" python -m core.train \
    --model tpinn --region "$REGION" --cv_mode loso --parametric_source --tau_s 5 \
    --init_seed "$INIT_SEED" --split_seed "$SPLIT_SEED" \
    --fold "$FOLD" --n_folds "$N_FOLDS" --epochs "$EPOCHS" \
    --part_dropout "$PART_DROPOUT" \
    --latent_dim "$LATENT_DIM" --embed_dim "$EMBED_DIM" --part_embed_dim "$PART_EMBED_DIM" \
    --beta_max "$BETA_MAX" --alpha "$ALPHA" --n_copies "$N_COPIES"
fi

PYTHONHASHSEED="$INIT_SEED" python -m core.tstr \
  --model tpinn --region "$REGION" --cv_mode loso --parametric_source --tau_s 5 \
  --init_seed "$INIT_SEED" --split_seed "$SPLIT_SEED" \
  --fold "$FOLD" --n_folds "$N_FOLDS" \
  --part_dropout "$PART_DROPOUT" \
  --latent_dim "$LATENT_DIM" --embed_dim "$EMBED_DIM" --alpha "$ALPHA" --n_copies "$N_COPIES" \
  --n_jobs 1 --eval_val --no_summary

# Aggregate after the array finishes: AGGREGATE=1 sbatch --array=0 experiments/28_tpinn_parametric_source_loso.sh
