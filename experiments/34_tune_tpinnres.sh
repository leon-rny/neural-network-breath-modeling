#!/bin/bash
#SBATCH --job-name=phys_tune
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --output=/home/rane10/logs/phystune.o%A_%a
#SBATCH --error=/home/rane10/logs/phystune.e%A_%a
#SBATCH --array=0-143%200
set -euo pipefail

source /opt/miniforge/etc/profile.d/conda.sh
conda activate nnbm
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# PHASE 4c — regularization/HP tuning of the physics CHAMPION (tpinn-res @ stdscale) on the active legacy-76s
# channel. tpinn-res ties cvae (KF 0.787/0.651 | LOSO 0.535/0.606) and arch combos (4b) added nothing; this is
# the last lever. We sweep the live HP axes (lambda_phys is MOOT for tpinn -> physics is the decoder, lambda=0):
#   latent_dim x beta_max x free_bits.  Committed center = ld16 / bm0.01 / fb0.0.
# Each grid point is namespaced via --hp_tag "bm{BM}_fb{FB}" (run_id suffix; ld already in run_id) so checkpoints
# never collide; results read back from results/tpinn/{run_id}_tstr.json. Search on a CHEAP kfold proxy
# (seeds 0,42 x folds 1,3), then VALIDATE the winner at the full protocol (kf 5x5 + loso 5x8) to avoid
# proxy-overfit (tuning-results memory: prior Optuna tuning did NOT beat the committed config).
#   CONFIG=search                              sbatch --array=0-143%200 experiments/34_tune_tpinnres.sh
#   LD=.. BM=.. FB=.. CONFIG=val_kfold         sbatch --array=0-49%200  experiments/34_tune_tpinnres.sh
#   LD=.. BM=.. FB=.. CONFIG=val_loso          sbatch --array=0-79%200  experiments/34_tune_tpinnres.sh
PHYS_PREP=stdscale; EMBED_DIM=8; PART_EMBED_DIM=8; ALPHA=0.05; N_COPIES=10
SPLIT_SEED=42; EPOCHS="${EPOCHS:-500}"; MFLAGS="--phys_residual"
LDS=(8 16 32); BMS=(0.003 0.01 0.03); FBS=(0.0 0.5)
REGIONS=(mouth nose)
CONFIG="${CONFIG:?set CONFIG=search|val_kfold|val_loso}"

run_one () {  # args: REGION LD BM FB CV PD INIT_SEED FOLD NF
  local REGION=$1 LD=$2 BM=$3 FB=$4 CV=$5 PD=$6 INIT_SEED=$7 FOLD=$8 NF=$9
  local HPTAG="bm${BM}_fb${FB}"
  local PARAMS="results/pinn/params_${REGION}.npy"
  [ -f "$PARAMS" ] || { echo "[PHYSTUNE] ERROR: missing $PARAMS"; exit 1; }
  echo "[PHYSTUNE] $CONFIG region=$REGION ld=$LD bm=$BM fb=$FB cv=$CV seed=$INIT_SEED fold=$FOLD hp=$HPTAG"
  if [ "${SKIP_TRAIN:-0}" != "1" ]; then
  PYTHONHASHSEED="$INIT_SEED" python -m core.train --model tpinn --region "$REGION" --cv_mode "$CV" \
    --phys_prep "$PHYS_PREP" --init_seed "$INIT_SEED" --split_seed "$SPLIT_SEED" --fold "$FOLD" --n_folds "$NF" \
    --epochs "$EPOCHS" --part_dropout "$PD" --latent_dim "$LD" --embed_dim "$EMBED_DIM" \
    --part_embed_dim "$PART_EMBED_DIM" --beta_max "$BM" --free_bits "$FB" --alpha "$ALPHA" --n_copies "$N_COPIES" \
    --hp_tag "$HPTAG" $MFLAGS
  fi
  PYTHONHASHSEED="$INIT_SEED" python -m core.tstr --model tpinn --region "$REGION" --cv_mode "$CV" \
    --phys_prep "$PHYS_PREP" --init_seed "$INIT_SEED" --split_seed "$SPLIT_SEED" --fold "$FOLD" --n_folds "$NF" \
    --part_dropout "$PD" --latent_dim "$LD" --embed_dim "$EMBED_DIM" --part_embed_dim "$PART_EMBED_DIM" \
    --free_bits "$FB" --alpha "$ALPHA" --n_copies "$N_COPIES" --hp_tag "$HPTAG" $MFLAGS \
    --n_jobs 1 --eval_val --no_summary
}

IDX=${SLURM_ARRAY_TASK_ID:-${TASK_ID:?set SLURM_ARRAY_TASK_ID or TASK_ID}}

if [ "$CONFIG" = search ]; then
  SEEDS_S=(0 42); FOLDS_S=(1 3); NF=5; CV=kfold; PD=0.0
  NG=$(( ${#LDS[@]} * ${#BMS[@]} * ${#FBS[@]} ))         # 18
  GRID_IDX=$(( IDX % NG ));            IDX=$(( IDX / NG ))
  FOLD_IDX=$(( IDX % ${#FOLDS_S[@]} ));IDX=$(( IDX / ${#FOLDS_S[@]} ))
  SEED_IDX=$(( IDX % ${#SEEDS_S[@]} ));IDX=$(( IDX / ${#SEEDS_S[@]} ))
  REGION_IDX=$(( IDX % ${#REGIONS[@]} ))
  FB_IDX=$(( GRID_IDX % ${#FBS[@]} )); G=$(( GRID_IDX / ${#FBS[@]} ))
  BM_IDX=$(( G % ${#BMS[@]} ));        G=$(( G / ${#BMS[@]} ))
  LD_IDX=$(( G % ${#LDS[@]} ))
  run_one "${REGIONS[$REGION_IDX]}" "${LDS[$LD_IDX]}" "${BMS[$BM_IDX]}" "${FBS[$FB_IDX]}" \
          "$CV" "$PD" "${SEEDS_S[$SEED_IDX]}" "${FOLDS_S[$FOLD_IDX]}" "$NF"
elif [ "$CONFIG" = val_kfold ] || [ "$CONFIG" = val_loso ]; then
  : "${LD:?set LD}"; : "${BM:?set BM}"; : "${FB:?set FB}"
  SEEDS=(0 1 7 42 123)
  if [ "$CONFIG" = val_loso ]; then
    CV=loso; PD=0.1
    NF=$(python -c "from core.data import load_dataset, n_loso_folds; print(n_loso_folds(load_dataset('dataset')))")
  else CV=kfold; PD=0.0; NF=5; fi
  FOLDS=($(seq 1 "$NF"))
  FOLD_IDX=$(( IDX % ${#FOLDS[@]} ));  IDX=$(( IDX / ${#FOLDS[@]} ))
  SEED_IDX=$(( IDX % ${#SEEDS[@]} ));  IDX=$(( IDX / ${#SEEDS[@]} ))
  REGION_IDX=$(( IDX % ${#REGIONS[@]} ))
  run_one "${REGIONS[$REGION_IDX]}" "$LD" "$BM" "$FB" "$CV" "$PD" \
          "${SEEDS[$SEED_IDX]}" "${FOLDS[$FOLD_IDX]}" "$NF"
else
  echo "bad CONFIG=$CONFIG"; exit 1
fi
