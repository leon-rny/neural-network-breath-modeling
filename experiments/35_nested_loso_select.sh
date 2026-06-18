#!/bin/bash
#SBATCH --job-name=loso_select
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --output=/home/rane10/logs/loso_select.o%A_%a
#SBATCH --error=/home/rane10/logs/loso_select.e%A_%a
#SBATCH --array=0-79%200
set -euo pipefail

source /opt/miniforge/etc/profile.d/conda.sh
conda activate nnbm

# loso axis data-driven; 2 regions x 5 seeds x N subjects, default N=8 -> --array=0-$((2*N*5-1))%200
# prerequisite: python -m ablations.loso shortlist --k 3
# then: sbatch experiments/35_nested_loso_select.sh ; then: python -m ablations.loso select
VARIANT=conv_baseline
PART_DROPOUT=0.1  # null-token required for loso generation
REGIONS=(mouth nose)
INIT_SEEDS=(0 1 7 42 123)
SPLIT_SEED=42
EPOCHS="${EPOCHS:-500}"

N_REGIONS=${#REGIONS[@]}
N_SEEDS=${#INIT_SEEDS[@]}

# single thread per task
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

N_DATA=$(python -c "from core.data import load_dataset, n_loso_folds; print(n_loso_folds(load_dataset('dataset')))")
N_SUBJECTS="$N_DATA"
if [ "$N_DATA" -lt 4 ]; then
  echo "[LOSO-A] ERROR: nested LOSO needs >=4 subjects (got $N_DATA); Stage A would train on <2. Record more subjects."; exit 1
fi

IDX=${SLURM_ARRAY_TASK_ID:-${TASK_ID:?set SLURM_ARRAY_TASK_ID (sbatch) or TASK_ID for a local run}}
SEED_IDX=$(( IDX % N_SEEDS )); IDX=$(( IDX / N_SEEDS ))
FOLD_IDX=$(( IDX % N_SUBJECTS )); IDX=$(( IDX / N_SUBJECTS ))
REGION_IDX=$(( IDX % N_REGIONS ))

REGION=${REGIONS[$REGION_IDX]}
INIT_SEED=${INIT_SEEDS[$SEED_IDX]}
T=$(( FOLD_IDX + 1 ))  # outer test subject (1-indexed)
V=$(( (T % N_SUBJECTS) + 1 ))  # val subject = neighbour of t

SHORTLIST="results/loso/shortlist_${REGION}.csv"
[ -f "$SHORTLIST" ] || { echo "[LOSO-A] ERROR: $SHORTLIST missing - run 'python -m ablations.loso shortlist' first."; exit 1; }

echo "[LOSO-A] region=$REGION outer_t=$T val_v=$V seed=$INIT_SEED | candidates from $SHORTLIST"
# shortlist rows: config,beta_max,accuracy (skip header)
tail -n +2 "$SHORTLIST" | while IFS=, read -r CONFIG BETA_MAX _REST; do
  [ -z "$CONFIG" ] && continue
  echo "[LOSO-A]   train(exclude=$T, test=$V) config=$CONFIG beta_max=$BETA_MAX"
  PYTHONHASHSEED="$INIT_SEED" python -m ablations.cvae --mode jittering \
    --variant "$VARIANT" --config "$CONFIG" --beta_max "$BETA_MAX" \
    --part_dropout "$PART_DROPOUT" \
    --cv_mode loso --loso_trial_val --loso_exclude "$T" \
    --fold "$V" --region "$REGION" \
    --init_seed "$INIT_SEED" --split_seed "$SPLIT_SEED" --n_folds "$N_SUBJECTS" \
    --epochs "$EPOCHS" --n_jobs 1 --skip_existing --no_summary
done

# then: python -m ablations.loso select ; then: sbatch --dependency=afterok:<thisjob> experiments/36_nested_loso_final.sh
