#!/bin/bash
#SBATCH --job-name=pinn
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mail-type=FAIL
#SBATCH --account=sc-users
#SBATCH --output=/home/rane10/logs/pinn.o%j
#SBATCH --error=/home/rane10/logs/pinn.e%j

set -euo pipefail

echo "Start time: $(date)"

cd ~/nnbm
source /opt/miniforge/etc/profile.d/conda.sh
conda activate toyenv

# Keep PyTorch/CUDA deterministic runs compatible with CuBLAS on GPU.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Stream Python logs immediately even when piped through tee.
export PYTHONUNBUFFERED=1

REGION="mouth"
SEED=42
export PYTHONHASHSEED=$SEED

export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

python -u train_pinn_stage2.py --train-only --region "$REGION" --seed "$SEED" 2>&1 | tee output.out

echo "End time: $(date)"
