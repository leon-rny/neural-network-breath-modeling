#!/bin/bash
set -euo pipefail

REGIONS=(nose)
SEEDS=(0 1 7 42 123)

LATENT_DIM=16
FREE_BITS=0.26488726939708584
BETA_MAX=0.015639543803971975
LR=0.008079753130669148
BATCH_SIZE=16
BETA_WARMUP_EPOCHS=179
EMBED_DIM=4

start_time=$(date +%s)
for region in "${REGIONS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    export PYTHONHASHSEED=$seed
    python -m core.train --model cvae --region "$region" --seed "$seed" --latent_dim "$LATENT_DIM" --free_bits "$FREE_BITS" --beta_max "$BETA_MAX" --lr "$LR" --batch_size "$BATCH_SIZE" --beta_warmup_epochs "$BETA_WARMUP_EPOCHS" --embed_dim "$EMBED_DIM"
    python -m core.tstr  --model cvae --region "$region" --seed "$seed" --latent_dim "$LATENT_DIM" --free_bits "$FREE_BITS" --embed_dim "$EMBED_DIM"
  done
done
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Total elapsed time: $elapsed seconds"
