#!/bin/bash
set -euo pipefail

REGIONS=(mouth nose)
SEEDS=(0 1 7 42 123)
VARIANTS=(conv_baseline conv_slim mlp mlp_small conv_asym)

start_time=$(date +%s)
for region in "${REGIONS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    export PYTHONHASHSEED=$seed
    for variant in "${VARIANTS[@]}"; do
      python -m ablations.cvae_architecture --variant "$variant" --region "$region" --seed "$seed"
    done
  done
done
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Total elapsed time: $elapsed seconds"
