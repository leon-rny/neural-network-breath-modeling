#!/bin/bash
#SBATCH --job-name=loso_pinn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --output=/home/rane10/logs/loso_pinn.o%A_%a
#SBATCH --error=/home/rane10/logs/loso_pinn.e%A_%a
#SBATCH --array=0-479%200
set -euo pipefail

source /opt/miniforge/etc/profile.d/conda.sh
conda activate nnbm

# PINN under LEAVE-ONE-SUBJECT-OUT. Sweeps lambda_phys (incl. 0.0 = no-physics control) so the
# same run isolates the effect of the physics constraint on cross-subject generalization.
# 6 lambda_phys x 2 regions x N SUBJECTS (from data) x 5 seeds; default N=8 -> 480 -> --array=0-479
# Compare against the unconstrained CVAE LOSO gap (Round 3): mouth -0.116, nose -0.057.
LATENT_DIM=16
EMBED_DIM=8
PART_EMBED_DIM=8
BETA_MAX=0.01
ALPHA=0.05
N_COPIES=10
PART_DROPOUT=0.1                # null-token participant dropout — REQUIRED for LOSO generation of
                               # the unseen held-out subject (matches the CVAE LOSO protocol, 07/08)
LAMBDA_PHYS=(0.0 0.001 0.005 0.01 0.05 0.1)
REGIONS=(mouth nose)
INIT_SEEDS=(0 1 7 42 123)
SPLIT_SEED=42
# subject count is data-driven; FOLDS = 1..N held-out subjects
N_FOLDS=$(python -c "from core.data import load_dataset, n_loso_folds; print(n_loso_folds(load_dataset('dataset')))")
FOLDS=($(seq 1 "$N_FOLDS"))    # 1-indexed held-out subject in [1, N_FOLDS]
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
  echo "[LOSO-PINN] aggregate-only: merging per-combo LOSO TSTR results into results/summary.csv"
  for lp in "${LAMBDA_PHYS[@]}"; do
    python -m core.tstr --aggregate --model pinn --cv_mode loso \
      --regions "$(IFS=, ; echo "${REGIONS[*]}")" \
      --init_seeds "$(IFS=, ; echo "${INIT_SEEDS[*]}")" \
      --folds "$(IFS=, ; echo "${FOLDS[*]}")" \
      --split_seed "$SPLIT_SEED" --n_folds "$N_FOLDS" \
      --latent_dim "$LATENT_DIM" --embed_dim "$EMBED_DIM" \
      --lambda_phys "$lp" --part_dropout "$PART_DROPOUT" --alpha "$ALPHA" --n_copies "$N_COPIES"
  done
  exit 0
fi

IDX=${SLURM_ARRAY_TASK_ID:-${TASK_ID:?set SLURM_ARRAY_TASK_ID (via sbatch) or TASK_ID for a local run}}
FOLD_IDX=$(( IDX % N_FOLDS_AX ));   IDX=$(( IDX / N_FOLDS_AX ))
SEED_IDX=$(( IDX % N_SEEDS ));      IDX=$(( IDX / N_SEEDS ))
REGION_IDX=$(( IDX % N_REGIONS ));  IDX=$(( IDX / N_REGIONS ))
LP_IDX=$(( IDX % N_LP ))

LAMBDA_PHYS_VAL=${LAMBDA_PHYS[$LP_IDX]}
REGION=${REGIONS[$REGION_IDX]}
INIT_SEED=${INIT_SEEDS[$SEED_IDX]}
FOLD=${FOLDS[$FOLD_IDX]}

echo "[LOSO-PINN] task=${SLURM_ARRAY_TASK_ID:-$TASK_ID} region=$REGION lambda_phys=$LAMBDA_PHYS_VAL init_seed=$INIT_SEED held_out_subject=$FOLD"

PARAMS="results/pinn/params_${REGION}.npy"
if [ ! -f "$PARAMS" ]; then
  echo "[LOSO-PINN] ERROR: missing CIR params ($PARAMS) — fit them before training the PINN." >&2
  exit 1
fi

# checkpoint name must match core.train's run_id (cv-marker _loso + drop-marker _drop<pd> after _f<fold>)
RUN_ID="${REGION}_s${INIT_SEED}_ld${LATENT_DIM}_ed${EMBED_DIM}_phys${LAMBDA_PHYS_VAL}_f${FOLD}_loso_drop${PART_DROPOUT}_a${ALPHA}_n${N_COPIES}"
CKPT="results/pinn/${RUN_ID}_checkpoint.pt"
if [ -f "$CKPT" ]; then
  echo "[LOSO-PINN] checkpoint exists, skipping training: $CKPT"
else
  PYTHONHASHSEED="$INIT_SEED" python -m core.train \
    --model pinn --region "$REGION" --cv_mode loso \
    --init_seed "$INIT_SEED" --split_seed "$SPLIT_SEED" \
    --fold "$FOLD" --n_folds "$N_FOLDS" \
    --lambda_phys "$LAMBDA_PHYS_VAL" --part_dropout "$PART_DROPOUT" \
    --latent_dim "$LATENT_DIM" --embed_dim "$EMBED_DIM" --part_embed_dim "$PART_EMBED_DIM" \
    --epochs "$EPOCHS" \
    --beta_max "$BETA_MAX" --alpha "$ALPHA" --n_copies "$N_COPIES"
fi

PYTHONHASHSEED="$INIT_SEED" python -m core.tstr \
  --model pinn --region "$REGION" --cv_mode loso \
  --init_seed "$INIT_SEED" --split_seed "$SPLIT_SEED" \
  --fold "$FOLD" --n_folds "$N_FOLDS" \
  --lambda_phys "$LAMBDA_PHYS_VAL" --part_dropout "$PART_DROPOUT" \
  --latent_dim "$LATENT_DIM" --embed_dim "$EMBED_DIM" --alpha "$ALPHA" --n_copies "$N_COPIES" \
  --n_jobs 1 --eval_val --no_summary

# Aggregate after the array finishes: AGGREGATE=1 sbatch --array=0 experiments/20_pinn_loso.sh
