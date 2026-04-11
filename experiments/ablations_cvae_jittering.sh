#!/bin/bash
set -euo pipefail

REGIONS=(mouth nose)
SEEDS=(0 1 7 42 123)
CONFIGS=(
  baseline
  a0.01_n2
  a0.01_n5
  a0.01_n10
  a0.05_n2
  a0.05_n5
  a0.05_n10
  a0.1_n2
  a0.1_n5
  a0.1_n10
)

start_time=$(date +%s)
for region in "${REGIONS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for config in "${CONFIGS[@]}"; do
      python -m ablations.cvae_jittering --config "$config" --region "$region" --seed "$seed"
    done
  done
done
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Total elapsed time: $elapsed seconds"
