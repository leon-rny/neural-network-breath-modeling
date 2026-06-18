#!/bin/bash
#SBATCH --job-name=trtr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --output=/home/rane10/logs/trtr.o%A_%a
#SBATCH --error=/home/rane10/logs/trtr.e%A_%a
#SBATCH --array=0-129%200
set -euo pipefail

source /opt/miniforge/etc/profile.d/conda.sh
conda activate nnbm

# [0..49] k-fold: 2 regions x 5 seeds x 5 folds = 50
# [50..]  loso: 2 regions x 5 seeds x N subjects (data-driven); default N=8 -> 0-129
# if N changes, submit with --array=0-$((50+2*5*N-1))%200
MODEL=trtr
REGIONS=(mouth nose)
INIT_SEEDS=(0 1 7 42 123)
KFOLDS=(1 2 3 4 5)
N_FOLDS=5
N_SUBJECTS=$(python -c "from core.data import load_dataset, n_loso_folds; print(n_loso_folds(load_dataset('dataset')))")
SPLIT_SEED=42
FORCE_REBUILD="${FORCE_REBUILD:-0}"

N_REGIONS=${#REGIONS[@]}
N_SEEDS=${#INIT_SEEDS[@]}
N_KFOLDS_AX=${#KFOLDS[@]}
N_KFOLD=$(( N_REGIONS * N_SEEDS * N_KFOLDS_AX ))
N_LOSO=$(( N_REGIONS * N_SEEDS * N_SUBJECTS ))

# single thread per task
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

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
  # k-fold
  FOLD_IDX=$(( IDX % N_KFOLDS_AX )); IDX=$(( IDX / N_KFOLDS_AX ))
  SEED_IDX=$(( IDX % N_SEEDS )); IDX=$(( IDX / N_SEEDS ))
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
  # loso (N_SUBJECTS data-driven, set at top)
  LIDX=$(( IDX - N_KFOLD ))
  FOLD_IDX=$(( LIDX % N_SUBJECTS )); LIDX=$(( LIDX / N_SUBJECTS ))
  SEED_IDX=$(( LIDX % N_SEEDS )); LIDX=$(( LIDX / N_SEEDS ))
  REGION_IDX=$(( LIDX % N_REGIONS ))
  REGION=${REGIONS[$REGION_IDX]}
  INIT_SEED=${INIT_SEEDS[$SEED_IDX]}
  T=$(( FOLD_IDX + 1 ))

  echo "[TRTR] task=${SLURM_ARRAY_TASK_ID:-$TASK_ID} loso region=$REGION init_seed=$INIT_SEED test_subj=$T force_rebuild=$FORCE_REBUILD"
  PYTHONHASHSEED="$INIT_SEED" python -m core.tstr \
    --model "$MODEL" --region "$REGION" \
    --cv_mode loso --loso_trial_val \
    --init_seed "$INIT_SEED" --split_seed "$SPLIT_SEED" \
    --fold "$T" --n_folds "$N_SUBJECTS" \
    --n_jobs 1 --no_summary $REBUILD
fi

# Aggregate after the array finishes: AGGREGATE=1 sbatch --array=0 experiments/02_trtr.sh
