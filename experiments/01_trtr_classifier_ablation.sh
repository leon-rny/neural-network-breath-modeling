#!/bin/bash
#SBATCH --job-name=ablations_trtr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --output=/home/rane10/logs/ablations_trtr.o%A_%a
#SBATCH --error=/home/rane10/logs/ablations_trtr.e%A_%a
#SBATCH --array=0-259%50
set -euo pipefail

source /opt/miniforge/etc/profile.d/conda.sh
conda activate nnbm

# [0..249] k-fold: 5 pipelines x 2 regions x 5 seeds x 5 folds = 250
# [250..259] legacy: replication pipeline, single 80/10/10 split, 2 reg x 5 seeds = 10

PIPELINES=(replication shap_fix lgbm_fix tsfresh_fix smote_fix)
REGIONS=(mouth nose)
INIT_SEEDS=(0 1 7 42 123)
FOLDS=(1 2 3 4 5)
SPLIT_SEED=42
N_FOLDS=5
FORCE_REBUILD="${FORCE_REBUILD:-0}"

N_PIPELINES=${#PIPELINES[@]}
N_REGIONS=${#REGIONS[@]}
N_SEEDS=${#INIT_SEEDS[@]}
N_FOLDS_AX=${#FOLDS[@]}
N_KFOLD=$(( N_PIPELINES * N_REGIONS * N_SEEDS * N_FOLDS_AX ))
N_LEGACY=$(( N_REGIONS * N_SEEDS ))

# single thread per task
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

if [ "${AGGREGATE:-0}" = "1" ]; then
  echo "[ABLATION] aggregate-only: merging results/ablation_trtr/*_trtr.json into summary.csv"
  python -m ablations.trtr --aggregate
  exit 0
fi

REBUILD=""
[ "$FORCE_REBUILD" = "1" ] && REBUILD="--force_rebuild"

IDX=${SLURM_ARRAY_TASK_ID:-${TASK_ID:?set SLURM_ARRAY_TASK_ID (via sbatch) or TASK_ID=<0..259> for a local run}}

if [ "$IDX" -lt "$N_KFOLD" ]; then
  FOLD_IDX=$(( IDX % N_FOLDS_AX )); IDX=$(( IDX / N_FOLDS_AX ))
  SEED_IDX=$(( IDX % N_SEEDS )); IDX=$(( IDX / N_SEEDS ))
  REGION_IDX=$(( IDX % N_REGIONS )); IDX=$(( IDX / N_REGIONS ))
  PIPELINE_IDX=$(( IDX % N_PIPELINES ))

  PIPELINE=${PIPELINES[$PIPELINE_IDX]}
  REGION=${REGIONS[$REGION_IDX]}
  INIT_SEED=${INIT_SEEDS[$SEED_IDX]}
  FOLD=${FOLDS[$FOLD_IDX]}

  echo "[ABLATION] task=${SLURM_ARRAY_TASK_ID:-$TASK_ID} kfold pipeline=$PIPELINE region=$REGION init_seed=$INIT_SEED fold=$FOLD force_rebuild=$FORCE_REBUILD"
  PYTHONHASHSEED="$INIT_SEED" python -m ablations.trtr \
    --region "$REGION" --pipeline "$PIPELINE" \
    --init_seed "$INIT_SEED" --split_seed "$SPLIT_SEED" \
    --fold "$FOLD" --n_folds "$N_FOLDS" \
    --n_jobs 1 --no_summary $REBUILD
else
  LIDX=$(( IDX - N_KFOLD ))
  SEED_IDX=$(( LIDX % N_SEEDS )); LIDX=$(( LIDX / N_SEEDS ))
  REGION_IDX=$(( LIDX % N_REGIONS ))

  REGION=${REGIONS[$REGION_IDX]}
  SEED=${INIT_SEEDS[$SEED_IDX]}

  echo "[ABLATION] task=${SLURM_ARRAY_TASK_ID:-$TASK_ID} legacy replication single_split region=$REGION seed=$SEED force_rebuild=$FORCE_REBUILD"
  PYTHONHASHSEED="$SEED" python -m ablations.trtr \
    --region "$REGION" --pipeline replication --single_split \
    --init_seed "$SEED" --split_seed "$SEED" \
    --n_jobs 1 --no_summary $REBUILD
fi

# Aggregate after the array finishes: AGGREGATE=1 sbatch --array=0 experiments/ablations_trtr.sh
