#!/bin/bash
set -euo pipefail

REGIONS=(nose)
INIT_SEEDS=(0 1 7 42 123)
FOLDS=(1 2 3 4 5)
SPLIT_SEED=42
N_FOLDS=5
export SPLIT_SEED N_FOLDS
SKIP_TRAIN="${SKIP_TRAIN:-0}"

# tuned cvae hyperparameters
LATENT_DIM=16
EMBED_DIM=4
FREE_BITS=0.26488726939708584
BETA_MAX=0.015639543803971975
LR=0.008079753130669148
BATCH_SIZE=16
BETA_WARMUP_EPOCHS=179
export LATENT_DIM EMBED_DIM FREE_BITS BETA_MAX LR BATCH_SIZE BETA_WARMUP_EPOCHS

# parallelism
JOBS="${JOBS:-8}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# build configs
COMBOS=()
for region in "${REGIONS[@]}"; do
  for init_seed in "${INIT_SEEDS[@]}"; do
    for fold in "${FOLDS[@]}"; do
      COMBOS+=("$region $init_seed $fold")
    done
  done
done
echo "[CVAE] ${#COMBOS[@]} combos | $JOBS parallel | 1 thread/job"

start_time=$(date +%s)

# phase 1: training in parallel
if [ "$SKIP_TRAIN" = "1" ]; then
  echo "[CVAE] phase 1/2: training SKIPPED (SKIP_TRAIN=1)"
else
  echo "[CVAE] phase 1/2: training"
  set +e
  printf '%s\n' "${COMBOS[@]}" | xargs -P "$JOBS" -I{} bash -c '
    read -r region init_seed fold <<<"$1"
    echo "[TRAIN] start  r=$region is=$init_seed f=$fold"
    PYTHONHASHSEED="$init_seed" python -m core.train \
      --model cvae --region "$region" \
      --init_seed "$init_seed" --split_seed "$SPLIT_SEED" \
      --fold "$fold" --n_folds "$N_FOLDS" \
      --latent_dim "$LATENT_DIM" --embed_dim "$EMBED_DIM" \
      --free_bits "$FREE_BITS" --beta_max "$BETA_MAX" \
      --lr "$LR" --batch_size "$BATCH_SIZE" \
      --beta_warmup_epochs "$BETA_WARMUP_EPOCHS"
  ' _ {}
  train_status=$?
  set -e
  [ $train_status -ne 0 ] && echo "[CVAE] WARNING: $train_status from training phase (some runs may have failed)"
fi

# phase 2: tstr sequentially
echo "[CVAE] phase 2/2: tstr"
for combo in "${COMBOS[@]}"; do
  read -r region init_seed fold <<<"$combo"
  PYTHONHASHSEED="$init_seed" python -m core.tstr \
    --model cvae --region "$region" \
    --init_seed "$init_seed" --split_seed "$SPLIT_SEED" \
    --fold "$fold" --n_folds "$N_FOLDS" \
    --latent_dim "$LATENT_DIM" --embed_dim "$EMBED_DIM" \
    --free_bits "$FREE_BITS"
done

end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Total elapsed time: $elapsed seconds"
