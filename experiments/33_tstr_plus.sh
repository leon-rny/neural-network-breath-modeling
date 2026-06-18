#!/bin/bash
#SBATCH --job-name=tstrplus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=06:00:00
#SBATCH --output=/home/rane10/logs/tstrplus.o%A
#SBATCH --error=/home/rane10/logs/tstrplus.e%A
set -euo pipefail

source /opt/miniforge/etc/profile.d/conda.sh
conda activate nnbm
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# TSTR+ augmentation value: does synthetic+real beat real-only? Train the committed cvae_part, then
# tstr_plus (real+synthetic classifier) at several augmentation ratios; compare to trtr (real-only).
# Single sequential task (folds x ratios) to avoid summary.csv write races.
COMMON="--model cvae_part --region mouth --init_seed 0 --split_seed 42 --n_folds 5 --latent_dim 16 --embed_dim 8 --part_embed_dim 8 --free_bits 0.0 --alpha 0.05 --n_copies 10"
for F in 1 2 3; do
  PYTHONHASHSEED=0 python -m core.train $COMMON --fold "$F" --epochs 500 --beta_max 0.01
  for R in 0.5 1.0 2.0; do
    echo "==TSTRPLUS_RESULT fold=$F ratio=$R=="
    PYTHONHASHSEED=0 python -m core.tstr $COMMON --fold "$F" --mode tstr_plus --augmentation_ratio "$R" --n_jobs 1 2>&1 | grep -iE "accuracy|augment" | tail -3
  done
done
echo "==TSTRPLUS_DONE=="
