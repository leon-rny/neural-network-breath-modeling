#!/bin/bash
set -euo pipefail

REGIONS=(mouth)
SEEDS=(0 1 7 42 123)
AUGMENTATION_RATIOS=(0.1 0.25 0.5 2.0 4.0 10.0)

start_time=$(date +%s)
for region in "${REGIONS[@]}"; do
  for augmentation_ratio in "${AUGMENTATION_RATIOS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      python -m core.tstr --model cvae --region "$region" --seed "$seed" --mode tstr_plus --augmentation_ratio "$augmentation_ratio"
    done
  done
done
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Total elapsed time: $elapsed seconds"
