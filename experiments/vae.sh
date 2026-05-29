#!/bin/bash
set -euo pipefail

REGIONS=(mouth nose)
INIT_SEEDS=(0 1 7 42 123)
FOLDS=(1 2 3 4 5)
FREE_BITS=(0.0 0.1 2.0)
SPLIT_SEED=42
N_FOLDS=5
export SPLIT_SEED N_FOLDS
SKIP_TRAIN="${SKIP_TRAIN:-0}"

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
  for fb in "${FREE_BITS[@]}"; do
    for init_seed in "${INIT_SEEDS[@]}"; do
      for fold in "${FOLDS[@]}"; do
        COMBOS+=("$region $init_seed $fold $fb")
      done
    done
  done
done
echo "[VAE] ${#COMBOS[@]} combos | $JOBS parallel | 1 thread/job"

start_time=$(date +%s)

# phase 1: training in parallel
if [ "$SKIP_TRAIN" = "1" ]; then
  echo "[VAE] phase 1/2: training SKIPPED (SKIP_TRAIN=1)"
else
  echo "[VAE] phase 1/2: training"
  set +e
  printf '%s\n' "${COMBOS[@]}" | xargs -P "$JOBS" -I{} bash -c '
    read -r region init_seed fold fb <<<"$1"
    echo "[TRAIN] start  r=$region is=$init_seed f=$fold fb=$fb"
    PYTHONHASHSEED="$init_seed" python -m core.train \
      --model vae --region "$region" \
      --init_seed "$init_seed" --split_seed "$SPLIT_SEED" \
      --fold "$fold" --n_folds "$N_FOLDS" \
      --free_bits "$fb"
  ' _ {}
  train_status=$?
  set -e
  [ $train_status -ne 0 ] && echo "[VAE] WARNING: $train_status from training phase (some runs may have failed)"
fi

# phase 2: tstr sequentially
echo "[VAE] phase 2/2: tstr"
for combo in "${COMBOS[@]}"; do
  read -r region init_seed fold fb <<<"$combo"
  PYTHONHASHSEED="$init_seed" python -m core.tstr \
    --model vae --region "$region" \
    --init_seed "$init_seed" --split_seed "$SPLIT_SEED" \
    --fold "$fold" --n_folds "$N_FOLDS" \
    --free_bits "$fb"
done

end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Total elapsed time: $elapsed seconds"
