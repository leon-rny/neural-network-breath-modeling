#!/bin/bash
set -euo pipefail

REGIONS=(nose mouth)
INIT_SEEDS=(0 1 7 42 123)
FOLDS=(1 2 3 4 5)
SPLIT_SEED=42
N_FOLDS=5
export SPLIT_SEED N_FOLDS
SKIP_TRAIN="${SKIP_TRAIN:-0}"

export LATENT_DIM=16
export EMBED_DIM=8
region_cfg() {
  case "$1" in
    mouth) echo "0 1 0.1" ;;
    nose)  echo "0.05 10 0.01" ;;
    *) echo "[SWEEP] ERROR: unknown region '$1'" >&2; return 1 ;;
  esac
}
export -f region_cfg

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
echo "[SWEEP] ${#COMBOS[@]} combos | $JOBS parallel | 1 thread/job"

start_time=$(date +%s)

# phase 1: training in parallel
if [ "$SKIP_TRAIN" = "1" ]; then
  echo "[SWEEP] phase 1/2: training SKIPPED (SKIP_TRAIN=1)"
else
  echo "[SWEEP] phase 1/2: training"
  set +e
  printf '%s\n' "${COMBOS[@]}" | xargs -P "$JOBS" -I{} bash -c '
    read -r region init_seed fold <<<"$1"
    read -r alpha n_copies beta_max <<<"$(region_cfg "$region")"
    echo "[TRAIN] start  r=$region is=$init_seed f=$fold a=$alpha n=$n_copies bmax=$beta_max"
    PYTHONHASHSEED="$init_seed" python -m core.train \
      --model cvae_part --region "$region" \
      --init_seed "$init_seed" --split_seed "$SPLIT_SEED" \
      --fold "$fold" --n_folds "$N_FOLDS" \
      --latent_dim "$LATENT_DIM" --embed_dim "$EMBED_DIM" \
      --alpha "$alpha" --n_copies "$n_copies" --beta_max "$beta_max"
  ' _ {}
  train_status=$?
  set -e
  [ $train_status -ne 0 ] && echo "[SWEEP] WARNING: $train_status from training phase (some runs may have failed)"
fi

# phase 2: tstr sequentially
echo "[SWEEP] phase 2/2: tstr"
for combo in "${COMBOS[@]}"; do
  read -r region init_seed fold <<<"$combo"
  # beta_max is training-only (not a tstr arg, not in the run_id) -> discard
  read -r alpha n_copies _beta_max <<<"$(region_cfg "$region")"
  PYTHONHASHSEED="$init_seed" python -m core.tstr \
    --model cvae_part --region "$region" \
    --init_seed "$init_seed" --split_seed "$SPLIT_SEED" \
    --fold "$fold" --n_folds "$N_FOLDS" \
    --latent_dim "$LATENT_DIM" --embed_dim "$EMBED_DIM" \
    --alpha "$alpha" --n_copies "$n_copies"
done

end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Total elapsed time: $elapsed seconds"
