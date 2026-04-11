#!/bin/bash
set -euo pipefail

REGIONS=(mouth nose)
SEEDS=(0 1 7 42 123)
CONFIGS=(
  beta_cap_0.001
  beta_cap_0.01
  beta_cap_0.1
  beta_cap_0.5
  beta_cap_1.0
  lag_5_100
  lag_5_250
  lag_10_100
  lag_10_250
  lag_5_250_beta_0.1
  lag_5_250_beta_0.01
)

start_time=$(date +%s)
for region in "${REGIONS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for config in "${CONFIGS[@]}"; do
      python -m ablations.cvae_training_dynamics --config "$config" --region "$region" --seed "$seed"
    done
  done
done
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Total elapsed time: $elapsed seconds"
