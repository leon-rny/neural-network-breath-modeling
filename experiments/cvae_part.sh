#!/bin/bash
set -euo pipefail

REGIONS=(nose)
SEEDS=(0 1 7 42 123)
LATENT_DIM=16
EMBED_DIM=8

start_time=$(date +%s)
for region in "${REGIONS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    export PYTHONHASHSEED=$seed
    python -m core.train --model cvae_part --region "$region" --seed "$seed" --latent_dim "$LATENT_DIM" --embed_dim "$EMBED_DIM" --beta_max 0.01 --alpha 0.05 --n_copies 10
    python -m core.tstr  --model cvae_part --region "$region" --seed "$seed" --latent_dim "$LATENT_DIM" --embed_dim "$EMBED_DIM"
  done
done
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Total elapsed time: $elapsed seconds"
 