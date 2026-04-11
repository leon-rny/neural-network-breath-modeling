#!/bin/bash
set -euo pipefail

REGIONS=(mouth nose)
SEEDS=(0 1 7 42 123)
PIPELINES=(original shap_fix split_fix full)

start_time=$(date +%s)
for pipeline in "${PIPELINES[@]}"; do
  for region in "${REGIONS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      python -m ablations.trtr \
        --region "$region" \
        --pipeline "$pipeline" \
        --seed "$seed" \
        --force_rebuild
    done
  done
done
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Total elapsed time: $elapsed seconds"
