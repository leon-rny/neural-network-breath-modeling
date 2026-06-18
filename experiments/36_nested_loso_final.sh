#!/bin/bash
#SBATCH --job-name=loso_final
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --output=/home/rane10/logs/loso_final.o%A_%a
#SBATCH --error=/home/rane10/logs/loso_final.e%A_%a
#SBATCH --array=0-79%200
set -euo pipefail

source /opt/miniforge/etc/profile.d/conda.sh
conda activate nnbm

# loso axis data-driven; 2 regions x 5 seeds x N subjects, default N=8 -> --array=0-$((2*N*5-1))%200
# prerequisite: experiments/35_nested_loso_select.sh finished and python -m ablations.loso select ran
# then: sbatch --dependency=afterok:<35-jobid> experiments/36_nested_loso_final.sh
VARIANT=conv_baseline
PART_DROPOUT=0.1
REGIONS=(mouth nose)
INIT_SEEDS=(0 1 7 42 123)
SPLIT_SEED=42
EPOCHS="${EPOCHS:-500}"

N_REGIONS=${#REGIONS[@]}
N_SEEDS=${#INIT_SEEDS[@]}

# single thread per task
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

if [ "${AGGREGATE:-0}" = "1" ]; then
  echo "[LOSO-B] aggregate-only: consolidating Stage-B conv_baseline + trtr rows"
  python -m ablations.loso aggregate
  exit 0
fi

N_DATA=$(python -c "from core.data import load_dataset, n_loso_folds; print(n_loso_folds(load_dataset('dataset')))")
N_SUBJECTS="$N_DATA"
if [ "$N_DATA" -lt 4 ]; then
  echo "[LOSO-B] ERROR: nested LOSO needs >=4 subjects (got $N_DATA)."; exit 1
fi

IDX=${SLURM_ARRAY_TASK_ID:-${TASK_ID:?set SLURM_ARRAY_TASK_ID (sbatch) or TASK_ID for a local run}}
SEED_IDX=$(( IDX % N_SEEDS )); IDX=$(( IDX / N_SEEDS ))
FOLD_IDX=$(( IDX % N_SUBJECTS )); IDX=$(( IDX / N_SUBJECTS ))
REGION_IDX=$(( IDX % N_REGIONS ))

REGION=${REGIONS[$REGION_IDX]}
INIT_SEED=${INIT_SEEDS[$SEED_IDX]}
T=$(( FOLD_IDX + 1 ))

SELECTED="results/loso/selected_${REGION}.csv"
[ -f "$SELECTED" ] || { echo "[LOSO-B] ERROR: $SELECTED missing - run 'python -m ablations.loso select' first."; exit 1; }
# selected rows: outer_fold,config,beta_max,accuracy
LINE=$(awk -F, -v t="$T" 'NR>1 && $1==t {print $2","$3}' "$SELECTED")
[ -n "$LINE" ] || { echo "[LOSO-B] ERROR: no selected config for outer_fold=$T in $SELECTED"; exit 1; }
CONFIG=${LINE%%,*}
BETA_MAX=${LINE##*,}

echo "[LOSO-B] region=$REGION test_t=$T seed=$INIT_SEED | selected config=$CONFIG beta_max=$BETA_MAX"

# conv_baseline: retrain on n-1, score held-out t
PYTHONHASHSEED="$INIT_SEED" python -m ablations.cvae --mode jittering \
  --variant "$VARIANT" --config "$CONFIG" --beta_max "$BETA_MAX" \
  --part_dropout "$PART_DROPOUT" \
  --cv_mode loso --loso_trial_val \
  --fold "$T" --region "$REGION" \
  --init_seed "$INIT_SEED" --split_seed "$SPLIT_SEED" --n_folds "$N_SUBJECTS" \
  --epochs "$EPOCHS" --n_jobs 1 --skip_existing --no_summary

# trtr baseline on the identical stage-b split (no generator, no config)
PYTHONHASHSEED="$INIT_SEED" python -m core.tstr --model trtr \
  --cv_mode loso --loso_trial_val \
  --fold "$T" --region "$REGION" \
  --init_seed "$INIT_SEED" --split_seed "$SPLIT_SEED" --n_folds "$N_SUBJECTS" \
  --n_jobs 1 --no_summary

# Aggregate after the array finishes: AGGREGATE=1 sbatch --array=0 experiments/36_nested_loso_final.sh
