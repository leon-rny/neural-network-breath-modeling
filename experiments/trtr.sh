#!/bin/bash
set -euo pipefail

REGIONS=(mouth nose)
SEEDS=(0 1 7 42 123)

start_time=$(date +%s)
for region in "${REGIONS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    python -m core.tstr --model trtr --region "$region" --seed "$seed" --force_rebuild
  done
done
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Total elapsed time: $elapsed seconds"
