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

# 8 optuna workers x 2 regions = 16 tasks (shared study per region)
MODEL="${MODEL:-cvae}"  # cvae | cvae_part | vae
REGIONS=(mouth nose)
WORKERS_PER_REGION=8
N_TRIALS_PER_WORKER="${N_TRIALS_PER_WORKER:-8}"  # 8 x 8 = 64 trials/region
SEARCH_EPOCHS="${SEARCH_EPOCHS:-200}"

# single thread per task
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

IDX=${SLURM_ARRAY_TASK_ID:-${TASK_ID:?set SLURM_ARRAY_TASK_ID or TASK_ID}}
REGION=${REGIONS[$(( IDX / WORKERS_PER_REGION ))]}
SAMPLER_SEED=$(( IDX ))  # distinct tpe seed per worker

echo "[TUNE] worker=$IDX model=$MODEL region=$REGION sampler_seed=$SAMPLER_SEED trials=$N_TRIALS_PER_WORKER epochs=$SEARCH_EPOCHS"
python -m ablations.tuning optuna --model "$MODEL" --region "$REGION" \
  --n_trials "$N_TRIALS_PER_WORKER" --epochs "$SEARCH_EPOCHS" \
  --folds 1 3 --seeds 0 42 --split_seed 42 --n_folds 5 \
  --n_jobs 1 --sampler_seed "$SAMPLER_SEED"

# next: validate the winner at full fidelity via experiments/38_tune_cvae_validate.sh
