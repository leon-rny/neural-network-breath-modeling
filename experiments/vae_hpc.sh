#!/bin/bash
#SBATCH --job-name=tune_vae
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --mail-type=FAIL
#SBATCH --account=sc-users
#SBATCH --output=/home/rane10/logs/tune_vae.o%j
#SBATCH --error=/home/rane10/logs/tune_vae.e%j

set -euo pipefail
echo "Start time: $(date)"

cd ~/nnbm
source /opt/miniforge/etc/profile.d/conda.sh
conda activate toyenv

# threads per trial — capped to avoid oversubscription when running parallel trials
THREADS_PER_TRIAL=4
N_PARALLEL=$((SLURM_CPUS_PER_TASK / THREADS_PER_TRIAL))

export OMP_NUM_THREADS=$THREADS_PER_TRIAL
export MKL_NUM_THREADS=$THREADS_PER_TRIAL
export OPENBLAS_NUM_THREADS=$THREADS_PER_TRIAL
export PYTHONUNBUFFERED=1

REGIONS=(mouth nose)
SEEDS=(0 1 7 42 123)

for region in "${REGIONS[@]}"; do
  # pre-build TRTR caches sequentially so parallel trials don't race on the same file
  for seed in "${SEEDS[@]}"; do
    python -m core.tstr --model trtr --region "$region" --seed "$seed"
  done

  python -m core.tuning \
    --model vae \
    --region "$region" \
    --n_trials 50 \
    --epochs 500 \
    --seeds "${SEEDS[@]}" \
    --n_jobs "$THREADS_PER_TRIAL" \
    --n_parallel "$N_PARALLEL" \
    --torch_threads "$THREADS_PER_TRIAL"
done

echo "End time: $(date)"
