#!/bin/bash
#SBATCH --job-name=tune_cvae_part
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --mail-type=FAIL
#SBATCH --account=sc-users
#SBATCH --output=/home/rane10/logs/tune_cvae.o%j
#SBATCH --error=/home/rane10/logs/tune_cvae.e%j

set -euo pipefail
mkdir -p ~/logs
echo "Start time: $(date)"

cd ~/nnbm
source /opt/miniforge/etc/profile.d/conda.sh
conda activate toyenv

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

REGIONS=(mouth nose)
SEEDS=(0 1 7 42 123)

for region in "${REGIONS[@]}"; do
  python -m core.tuning \
    --model cvae_part \
    --region "$region" \
    --n_trials 50 \
    --epochs 500 \
    --seeds "${SEEDS[@]}" \
    --n_jobs 4 \
    --torch_threads "$SLURM_CPUS_PER_TASK"
done

echo "End time: $(date)"
