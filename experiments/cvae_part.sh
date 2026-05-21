#!/bin/bash
set -euo pipefail

REGIONS=(nose mouth)
SEEDS=(0 1 7 42 123)
export LATENT_DIM=16
export EMBED_DIM=8

# Region-specific config: single source of truth used by BOTH training and
# tstr so the run_id (checkpoint path + summary row) always stays consistent.
#   echoes "<alpha> <n_copies> <beta_max>"
#   mouth: no jittering (alpha=0, n_copies=1 -> no run_id suffix); beta_max=0.1
#   nose : jitter alpha=0.05, n_copies=10;                         beta_max=0.01
region_cfg() {
  case "$1" in
    mouth) echo "0 1 0.1" ;;
    nose)  echo "0.05 10 0.01" ;;
    *) echo "[SWEEP] ERROR: unknown region '$1'" >&2; return 1 ;;
  esac
}
export -f region_cfg
# SKIP_TRAIN=1 runs phase 2 (tstr) only — use when checkpoints already exist.
SKIP_TRAIN="${SKIP_TRAIN:-0}"

# parallelism: each job pinned to 1 thread so JOBS of them share the
# cores without oversubscription. M1 Max has 10 cores; 8 leaves headroom.
JOBS="${JOBS:-8}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1   # Apple Accelerate (PyTorch CPU on Apple Silicon)
export NUMEXPR_NUM_THREADS=1

# build the (region, seed) job list
COMBOS=()
for region in "${REGIONS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    COMBOS+=("$region $seed")
  done
done
echo "[SWEEP] ${#COMBOS[@]} combos | $JOBS parallel | 1 thread/job"

start_time=$(date +%s)

# --- phase 1: training (parallel; each combo writes a unique run_id) ---
if [ "$SKIP_TRAIN" = "1" ]; then
  echo "[SWEEP] phase 1/2: training SKIPPED (SKIP_TRAIN=1)"
else
  echo "[SWEEP] phase 1/2: training"
  set +e
  printf '%s\n' "${COMBOS[@]}" | xargs -P "$JOBS" -I{} bash -c '
    read -r region seed <<<"$1"
    read -r alpha n_copies beta_max <<<"$(region_cfg "$region")"
    echo "[TRAIN] start  r=$region s=$seed a=$alpha n=$n_copies bmax=$beta_max"
    PYTHONHASHSEED="$seed" python -m core.train \
      --model cvae_part --region "$region" --seed "$seed" \
      --latent_dim "$LATENT_DIM" --embed_dim "$EMBED_DIM" \
      --alpha "$alpha" --n_copies "$n_copies" --beta_max "$beta_max"
  ' _ {}
  train_status=$?
  set -e
  [ $train_status -ne 0 ] && echo "[SWEEP] WARNING: $train_status from training phase (some runs may have failed)"
fi

# --- phase 2: tstr (sequential; save_summary() races on results/summary.csv) ---
echo "[SWEEP] phase 2/2: tstr"
for combo in "${COMBOS[@]}"; do
  read -r region seed <<<"$combo"
  # beta_max is training-only (not a tstr arg, not in the run_id) -> discard
  read -r alpha n_copies _beta_max <<<"$(region_cfg "$region")"
  PYTHONHASHSEED="$seed" python -m core.tstr \
    --model cvae_part --region "$region" --seed "$seed" \
    --latent_dim "$LATENT_DIM" --embed_dim "$EMBED_DIM" \
    --alpha "$alpha" --n_copies "$n_copies"
done

end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Total elapsed time: $elapsed seconds"
