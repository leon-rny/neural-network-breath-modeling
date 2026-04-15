#!/bin/bash
set -euo pipefail

REGIONS=(mouth)
SEEDS=(0 1 7 42 123)
LATENT_DIM=16
EMBED_DIM=8

start_time=$(date +%s)
for region in "${REGIONS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    export PYTHONHASHSEED=$seed
    python -m core.train --model cvae_part --region "$region" --seed "$seed" --latent_dim "$LATENT_DIM" --embed_dim "$EMBED_DIM" --beta_max 0.1
    python -m core.tstr  --model cvae_part --region "$region" --seed "$seed" --latent_dim "$LATENT_DIM" --embed_dim "$EMBED_DIM"
  done
done
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Total elapsed time: $elapsed seconds"
