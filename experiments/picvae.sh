#!/bin/bash
set -euo pipefail

REGIONS=(mouth)
SEEDS=(0 1 7 42 123)

start_time=$(date +%s)
for region in "${REGIONS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    python train_picvae.py --region "$region" --seed "$seed" --lambda_physics 0.001
    python -m core.tstr  --model picvae --region "$region" --seed "$seed" --lambda_physics 0.001
  done
done
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Total elapsed time: $elapsed seconds"
