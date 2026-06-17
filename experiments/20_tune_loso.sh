#!/bin/bash
#SBATCH --job-name=tune_loso
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=08:00:00
#SBATCH --output=/home/rane10/logs/tune_loso.o%A_%a
#SBATCH --error=/home/rane10/logs/tune_loso.e%A_%a
#SBATCH --array=0-15%16
set -euo pipefail

source /opt/miniforge/etc/profile.d/conda.sh
conda activate nnbm

# NESTED LOSO hyperparameter search (cross-subject TSTR). Tunes on a DEV subset of held-out subjects
# (folds {2,4} = subjects e,g) — each trains on the other 4 subjects and generates the held-out one
# via the null token (part_dropout=0.1). The disjoint subjects {a,f,p} (folds 1,3,5) are NEVER scored
# during the search, so validating the winner on them (19_tune_validate.sh CV_MODE=loso) is an unbiased
# cross-subject estimate. cvae_part only (needs the participant null token).
MODEL="${MODEL:-cvae_part}"
REGIONS=(mouth nose)
WORKERS_PER_REGION=8
N_TRIALS_PER_WORKER="${N_TRIALS_PER_WORKER:-8}"
SEARCH_EPOCHS="${SEARCH_EPOCHS:-200}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

IDX=${SLURM_ARRAY_TASK_ID:-${TASK_ID:?set SLURM_ARRAY_TASK_ID or TASK_ID}}
REGION=${REGIONS[$(( IDX / WORKERS_PER_REGION ))]}
SAMPLER_SEED=$(( IDX ))

echo "[TUNE-LOSO] worker=$IDX model=$MODEL region=$REGION sampler_seed=$SAMPLER_SEED dev_folds={2,4}=e,g"
python -m core.tuning --model "$MODEL" --region "$REGION" \
  --cv_mode loso --part_dropout 0.1 \
  --n_trials "$N_TRIALS_PER_WORKER" --epochs "$SEARCH_EPOCHS" \
  --folds 2 4 --seeds 0 42 --split_seed 42 --n_folds 5 \
  --n_jobs 1 --sampler_seed "$SAMPLER_SEED"

# best params -> results/tuning/<model>_<region>_loso_best.json ; validate on honest holdout {a,f,p}
# (folds 1,3,5) via: CV_MODE=loso MODEL=<m> REGION=<r> sbatch experiments/19_tune_validate.sh
