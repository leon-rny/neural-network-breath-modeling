#!/bin/bash
set -euo pipefail

SEEDS=(0 1 7 42 123)
REGIONS=('mouth' 'nose')
LAMBDA_PHYS=(0 0.001 0.005 0.01 0.05 0.1)
LATENT_DIM=16
EMBED_DIM=8

start_time=$(date +%s)
for lambda_phys in "${LAMBDA_PHYS[@]}"; do
  for region in "${REGIONS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      export PYTHONHASHSEED=$seed
      python train_pinn.py --region "$region" --seed "$seed" --lambda_phys "$lambda_phys" --latent_dim "$LATENT_DIM" --embed_dim "$EMBED_DIM"
      python -m core.tstr --model pinn --region "$region" --seed "$seed" --lambda_phys "$lambda_phys" --latent_dim "$LATENT_DIM" --embed_dim "$EMBED_DIM"
    done
  done
done
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Total elapsed time: $elapsed seconds"
