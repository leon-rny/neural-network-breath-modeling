#!/bin/bash 
set -euo pipefail

SEEDS=(0 1 7 42 123)
REGION=('mouth' 'nose')
LAMBDA_RESIDUAL=(0.01)

start_time=$(date +%s)
for lambda_residual in "${LAMBDA_RESIDUAL[@]}"; do
  for region in "${REGION[@]}"; do
    for seed in "${SEEDS[@]}"; do
      export PYTHONHASHSEED=$seed
      python -m core.tstr --model pinn --region $region --latent_dim 16 --embed_dim 8 --seed $seed --lambda_residual $lambda_residual
    done
  done
done
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Total elapsed time: $elapsed seconds"