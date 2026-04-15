#!/bin/bash
#SBATCH --job-name=cvae
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=25:00:00
#SBATCH --mail-type=FAIL
#SBATCH --account=sc-users
#SBATCH --output=/home/rane10/logs/cvae.o%j
#SBATCH --error=/home/rane10/logs/cvae.e%j

echo "Start time: $(date)"

REGIONS=(mouth)
SEEDS=(0 1 7 42 123)
LATENT_DIM=16
EMBED_DIM=8
BETA_MAX=0.1
N_JOBS=16

cd ~/nnbm
source /opt/miniforge/etc/profile.d/conda.sh
conda activate toyenv
for region in "${REGIONS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    export PYTHONHASHSEED=$seed
    python -m core.train --model cvae_part --region "$region" --seed "$seed" --latent_dim "$LATENT_DIM" --embed_dim "$EMBED_DIM" --beta_max "$BETA_MAX"
    python -m core.tstr  --model cvae_part --region "$region" --seed "$seed" --latent_dim "$LATENT_DIM" --embed_dim "$EMBED_DIM" --n_jobs "$N_JOBS"
  done
done

echo "End time: $(date)"
