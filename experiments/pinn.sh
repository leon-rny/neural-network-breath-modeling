#!/bin/bash
set -euo pipefail

SEEDS=(0 1 7 42 123)
REGION=('nose')

start_time=$(date +%s)
for region in "${REGION[@]}"; do
  for seed in "${SEEDS[@]}"; do
    export PYTHONHASHSEED=$seed
    python -m core.tstr --model pinn --region $region --latent_dim 16 --embed_dim 8 --seed $seed
  done
done
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Total elapsed time: $elapsed seconds"