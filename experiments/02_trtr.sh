#!/bin/bash
#SBATCH --job-name=trtr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --output=/home/rane10/logs/trtr.o%A_%a
#SBATCH --error=/home/rane10/logs/trtr.e%A_%a
#SBATCH --array=0-109%50
set -euo pipefail

source /opt/miniforge/etc/profile.d/conda.sh
conda activate nnbm

# TRTR (real-data ceiling) under BOTH protocols, in one array:
#   [0..49]   k-fold CV  : 2 regions x 5 seeds x 5 folds          = 50   -> within-subject ceiling
#   [50..109] LOSO       : 2 regions x 5 seeds x 6 subjects       = 60   -> cross-subject ceiling
# LOSO uses --loso_trial_val (train on n-1 subjects, score the held-out one) so it matches the
# conv_baseline Stage-B split and is apples-to-apples with the LOSO model numbers.
#
# NOTE for the 6-subject rerun: the k-fold TRTR cache path does NOT encode subject count, so clear
# stale caches once before running:  rm -f results/trtr/*.pkl   (or submit with FORCE_REBUILD=1).
MODEL=trtr
REGIONS=(mouth nose)
INIT_SEEDS=(0 1 7 42 123)
KFOLDS=(1 2 3 4 5)            # k-fold trial-folds
N_FOLDS=5                     # k-fold count
N_SUBJECTS=6                  # LOSO: one fold per subject (asserted against data for LOSO tasks)
SPLIT_SEED=42
FORCE_REBUILD="${FORCE_REBUILD:-0}"

N_REGIONS=${#REGIONS[@]}
N_SEEDS=${#INIT_SEEDS[@]}
N_KFOLDS_AX=${#KFOLDS[@]}
N_KFOLD=$(( N_REGIONS * N_SEEDS * N_KFOLDS_AX ))   # 50
N_LOSO=$(( N_REGIONS * N_SEEDS * N_SUBJECTS ))     # 60

# single thread per task
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [ "${AGGREGATE:-0}" = "1" ]; then
  echo "[TRTR] aggregate-only: merging k-fold + LOSO trtr rows into results/summary.csv"
  python -m core.tstr --aggregate --model "$MODEL" \
    --regions "$(IFS=, ; echo "${REGIONS[*]}")" \
    --init_seeds "$(IFS=, ; echo "${INIT_SEEDS[*]}")" \
    --folds "$(IFS=, ; echo "${KFOLDS[*]}")" \
    --split_seed "$SPLIT_SEED" --n_folds "$N_FOLDS"
  LOSO_FOLDS=$(seq -s, 1 "$N_SUBJECTS")
  python -m core.tstr --aggregate --model "$MODEL" --cv_mode loso --loso_trial_val \
    --regions "$(IFS=, ; echo "${REGIONS[*]}")" \
    --init_seeds "$(IFS=, ; echo "${INIT_SEEDS[*]}")" \
    --folds "$LOSO_FOLDS" \
    --split_seed "$SPLIT_SEED" --n_folds "$N_SUBJECTS"
  exit 0
fi

REBUILD=""
[ "$FORCE_REBUILD" = "1" ] && REBUILD="--force_rebuild"

IDX=${SLURM_ARRAY_TASK_ID:-${TASK_ID:?set SLURM_ARRAY_TASK_ID (via sbatch) or TASK_ID=<0..109> for a local run}}

if [ "$IDX" -lt "$N_KFOLD" ]; then
  # ---- k-fold regime ----
  FOLD_IDX=$(( IDX % N_KFOLDS_AX )); IDX=$(( IDX / N_KFOLDS_AX ))
  SEED_IDX=$(( IDX % N_SEEDS ));     IDX=$(( IDX / N_SEEDS ))
  REGION_IDX=$(( IDX % N_REGIONS ))
  REGION=${REGIONS[$REGION_IDX]}
  INIT_SEED=${INIT_SEEDS[$SEED_IDX]}
  FOLD=${KFOLDS[$FOLD_IDX]}

  echo "[TRTR] task=${SLURM_ARRAY_TASK_ID:-$TASK_ID} kfold region=$REGION init_seed=$INIT_SEED fold=$FOLD force_rebuild=$FORCE_REBUILD"
  PYTHONHASHSEED="$INIT_SEED" python -m core.tstr \
    --model "$MODEL" --region "$REGION" \
    --init_seed "$INIT_SEED" --split_seed "$SPLIT_SEED" \
    --fold "$FOLD" --n_folds "$N_FOLDS" \
    --n_jobs 1 --no_summary $REBUILD
else
  # ---- LOSO regime ----
  N_DATA=$(python -c "from core.data import load_dataset, n_loso_folds; print(n_loso_folds(load_dataset('dataset')))")
  if [ "$N_DATA" != "$N_SUBJECTS" ]; then
    echo "[TRTR] ERROR: dataset has $N_DATA subjects but LOSO sized for N_SUBJECTS=$N_SUBJECTS. Fix N_SUBJECTS and --array (=N_KFOLD + 2*N*5 - 1)."; exit 1
  fi
  LIDX=$(( IDX - N_KFOLD ))
  FOLD_IDX=$(( LIDX % N_SUBJECTS )); LIDX=$(( LIDX / N_SUBJECTS ))
  SEED_IDX=$(( LIDX % N_SEEDS ));    LIDX=$(( LIDX / N_SEEDS ))
  REGION_IDX=$(( LIDX % N_REGIONS ))
  REGION=${REGIONS[$REGION_IDX]}
  INIT_SEED=${INIT_SEEDS[$SEED_IDX]}
  T=$(( FOLD_IDX + 1 ))            # held-out test subject (1-indexed)

  echo "[TRTR] task=${SLURM_ARRAY_TASK_ID:-$TASK_ID} loso region=$REGION init_seed=$INIT_SEED test_subj=$T force_rebuild=$FORCE_REBUILD"
  PYTHONHASHSEED="$INIT_SEED" python -m core.tstr \
    --model "$MODEL" --region "$REGION" \
    --cv_mode loso --loso_trial_val \
    --init_seed "$INIT_SEED" --split_seed "$SPLIT_SEED" \
    --fold "$T" --n_folds "$N_SUBJECTS" \
    --n_jobs 1 --no_summary $REBUILD
fi

# Aggregate after the array finishes (merges both protocols into results/summary.csv, cv_mode column):
#   AGGREGATE=1 sbatch --array=0 experiments/02_trtr.sh
