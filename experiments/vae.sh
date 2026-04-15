#!/bin/bash
set -euo pipefail

REGIONS=(mouth nose)
SEEDS=(0 1 7 42 123)
FREE_BITS=(0.0 0.1 2.0)

start_time=$(date +%s)
for region in "${REGIONS[@]}"; do
  for fb in "${FREE_BITS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      export PYTHONHASHSEED=$seed
      python -m core.train --model vae --region "$region" --seed "$seed" --free_bits "$fb"
      python -m core.tstr  --model vae --region "$region" --seed "$seed" --free_bits "$fb"
    done
  done
done
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Total elapsed time: $elapsed seconds"
