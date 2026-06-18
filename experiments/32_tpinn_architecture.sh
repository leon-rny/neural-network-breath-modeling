#!/bin/bash
#SBATCH --job-name=phys_arch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --output=/home/rane10/logs/physarch.o%A_%a
#SBATCH --error=/home/rane10/logs/physarch.e%A_%a
#SBATCH --array=0-79%200
set -euo pipefail

source /opt/miniforge/etc/profile.d/conda.sh
conda activate nnbm
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# PHASE 4b "architecture combos" — tpinn-res is the one physics variant that, under --phys_prep stdscale,
# matches the CVAE (4a). Can ADDING transport-learnability on top of the residual push it ABOVE cvae?
# Two combos vs the plain-res baseline (fixed transport + residual):
#   ctres   = --class_transport --phys_residual  (per-class learnable D,v + residual; run_id _tphys_ct_res)
#   learnres= --learn_cir_params --phys_residual (global learnable D,v + residual; run_id _tphys_res_learn)
# ALL on stdscale + the ACTIVE (legacy-76s) channel. Compare to tpinn-res-stdscale (0.787/0.651 KF; 0.535/0.606 LOSO).
# Launch one CONFIG per array job (chain them); kfold -> --array=0-49%200, loso -> --array=0-79%200:
#   CONFIG=ctres_kfold|ctres_loso|learnres_kfold|learnres_loso
# Aggregate: CONFIG=<cfg> AGGREGATE=1 sbatch --array=0 experiments/32_tpinn_architecture.sh
PHYS_PREP="${PHYS_PREP:-stdscale}"
LATENT_DIM=16; EMBED_DIM=8; PART_EMBED_DIM=8; BETA_MAX=0.01; ALPHA=0.05; N_COPIES=10
REGIONS=(mouth nose); SEEDS=(0 1 7 42 123); SPLIT_SEED=42; EPOCHS="${EPOCHS:-500}"
CONFIG="${CONFIG:?set CONFIG=ctres_kfold|ctres_loso|learnres_kfold|learnres_loso}"
case "$CONFIG" in
  ctres_kfold)    MODEL=tpinn; CV=kfold; PD=0.0; MFLAGS="--class_transport --phys_residual" ;;
  ctres_loso)     MODEL=tpinn; CV=loso;  PD=0.1; MFLAGS="--class_transport --phys_residual" ;;
  learnres_kfold) MODEL=tpinn; CV=kfold; PD=0.0; MFLAGS="--learn_cir_params --phys_residual" ;;
  learnres_loso)  MODEL=tpinn; CV=loso;  PD=0.1; MFLAGS="--learn_cir_params --phys_residual" ;;
  *) echo "bad CONFIG=$CONFIG"; exit 1 ;;
esac
if [ "$CV" = loso ]; then
  NF=$(python -c "from core.data import load_dataset, n_loso_folds; print(n_loso_folds(load_dataset('dataset')))")
else NF=5; fi
FOLDS=($(seq 1 "$NF"))
N_REGIONS=${#REGIONS[@]}; N_SEEDS=${#SEEDS[@]}; N_FOLDS_AX=${#FOLDS[@]}

if [ "${AGGREGATE:-0}" = "1" ]; then
  echo "[PHYSARCH] aggregate $CONFIG (phys_prep=$PHYS_PREP)"
  python -m core.tstr --aggregate --model "$MODEL" --cv_mode "$CV" --phys_prep "$PHYS_PREP" \
    --regions "$(IFS=, ; echo "${REGIONS[*]}")" --init_seeds "$(IFS=, ; echo "${SEEDS[*]}")" \
    --folds "$(seq -s, 1 "$NF")" --split_seed "$SPLIT_SEED" --n_folds "$NF" \
    --latent_dim "$LATENT_DIM" --embed_dim "$EMBED_DIM" --part_embed_dim "$PART_EMBED_DIM" \
    --part_dropout "$PD" --alpha "$ALPHA" --n_copies "$N_COPIES" $MFLAGS
  exit 0
fi

IDX=${SLURM_ARRAY_TASK_ID:-${TASK_ID:?set SLURM_ARRAY_TASK_ID or TASK_ID}}
FOLD_IDX=$(( IDX % N_FOLDS_AX )); IDX=$(( IDX / N_FOLDS_AX ))
SEED_IDX=$(( IDX % N_SEEDS ));    IDX=$(( IDX / N_SEEDS ))
REGION_IDX=$(( IDX % N_REGIONS ))
REGION=${REGIONS[$REGION_IDX]}; INIT_SEED=${SEEDS[$SEED_IDX]}; FOLD=${FOLDS[$FOLD_IDX]}

PARAMS="results/pinn/params_${REGION}.npy"
[ -f "$PARAMS" ] || { echo "[PHYSARCH] ERROR: missing $PARAMS"; exit 1; }
echo "[PHYSARCH] $CONFIG region=$REGION seed=$INIT_SEED fold=$FOLD phys_prep=$PHYS_PREP"

if [ "${SKIP_TRAIN:-0}" != "1" ]; then
PYTHONHASHSEED="$INIT_SEED" python -m core.train --model "$MODEL" --region "$REGION" --cv_mode "$CV" \
  --phys_prep "$PHYS_PREP" --init_seed "$INIT_SEED" --split_seed "$SPLIT_SEED" --fold "$FOLD" --n_folds "$NF" \
  --epochs "$EPOCHS" --part_dropout "$PD" --latent_dim "$LATENT_DIM" --embed_dim "$EMBED_DIM" \
  --part_embed_dim "$PART_EMBED_DIM" --beta_max "$BETA_MAX" --alpha "$ALPHA" --n_copies "$N_COPIES" $MFLAGS
fi

PYTHONHASHSEED="$INIT_SEED" python -m core.tstr --model "$MODEL" --region "$REGION" --cv_mode "$CV" \
  --phys_prep "$PHYS_PREP" --init_seed "$INIT_SEED" --split_seed "$SPLIT_SEED" --fold "$FOLD" --n_folds "$NF" \
  --part_dropout "$PD" --latent_dim "$LATENT_DIM" --embed_dim "$EMBED_DIM" --part_embed_dim "$PART_EMBED_DIM" \
  --alpha "$ALPHA" --n_copies "$N_COPIES" $MFLAGS --n_jobs 1 --eval_val --no_summary
