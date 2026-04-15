#!/bin/bash
set -euo pipefail

REGIONS=(mouth nose)
SEEDS=(0 1 7 42 123)

start_time=$(date +%s)

# gan
for region in "${REGIONS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    export PYTHONHASHSEED=$seed
    python -m core.tstr --model gan --region "$region" --seed "$seed"
  done
done

# vae
for region in "${REGIONS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    export PYTHONHASHSEED=$seed
    python -m core.tstr --model vae --region "$region" --seed "$seed" --latent_dim 32 --free_bits 0.0
  done
done

# cvae
for seed in "${SEEDS[@]}"; do
  export PYTHONHASHSEED=$seed
  python -m core.tstr --model cvae --region mouth --seed "$seed" --latent_dim 32 --embed_dim 16 --free_bits 0.0
done

end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Total elapsed time: $elapsed seconds"
