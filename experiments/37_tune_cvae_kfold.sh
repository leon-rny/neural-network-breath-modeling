#!/bin/bash
#SBATCH --job-name=tune
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=06:00:00
#SBATCH --output=/home/rane10/logs/tune.o%A_%a
#SBATCH --error=/home/rane10/logs/tune.e%A_%a
#SBATCH --array=0-15%16
set -euo pipefail

source /opt/miniforge/etc/profile.d/conda.sh
conda activate nnbm

# Parallel Optuna TSTR search for the (c)VAE generators. All workers for a region share ONE study
# via the JournalFile backend (results/tuning/<model>_<region>_v4.log, load_if_exists), so they
# explore concurrently. 8 workers/region x N_TRIALS_PER_WORKER trials. SEARCH fidelity: 200 epochs,
# folds {1,3} x seeds {0,42} (whose trtr caches already exist -> no rebuild). Validate the winner
# afterwards at full 500ep x 5fold x 5seed (experiments/38_tune_cvae_validate.sh, generated from best.json).
MODEL="${MODEL:-cvae}"            # cvae | cvae_part | vae
REGIONS=(mouth nose)
WORKERS_PER_REGION=8
N_TRIALS_PER_WORKER="${N_TRIALS_PER_WORKER:-8}"   # 8 x 8 = 64 trials/region
SEARCH_EPOCHS="${SEARCH_EPOCHS:-200}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

IDX=${SLURM_ARRAY_TASK_ID:-${TASK_ID:?set SLURM_ARRAY_TASK_ID or TASK_ID}}
REGION=${REGIONS[$(( IDX / WORKERS_PER_REGION ))]}
SAMPLER_SEED=$(( IDX ))           # distinct TPE seed per worker -> diverse proposals

echo "[TUNE] worker=$IDX model=$MODEL region=$REGION sampler_seed=$SAMPLER_SEED trials=$N_TRIALS_PER_WORKER epochs=$SEARCH_EPOCHS"
python -m ablations.tuning optuna --model "$MODEL" --region "$REGION" \
  --n_trials "$N_TRIALS_PER_WORKER" --epochs "$SEARCH_EPOCHS" \
  --folds 1 3 --seeds 0 42 --split_seed 42 --n_folds 5 \
  --n_jobs 1 --sampler_seed "$SAMPLER_SEED"

# After the array finishes, read results/tuning/<model>_<region>_best.json and validate the winner
# at full fidelity (5 folds x 5 seeds, 500 epochs) before reporting.
