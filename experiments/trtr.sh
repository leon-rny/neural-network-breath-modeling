#!/bin/bash
set -euo pipefail

REGIONS=(mouth nose)
INIT_SEEDS=(0 1 7 42 123)
FOLDS=(1 2 3 4 5)
SPLIT_SEED=42
N_FOLDS=5
export SPLIT_SEED N_FOLDS
SKIP_BUILD="${SKIP_BUILD:-0}"

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
echo "[TRTR] ${#COMBOS[@]} combos | $JOBS parallel | 1 thread/job"

start_time=$(date +%s)

# phase 1: build caches in parallel
if [ "$SKIP_BUILD" = "1" ]; then
  echo "[TRTR] phase 1/2: build SKIPPED (SKIP_BUILD=1)"
else
  echo "[TRTR] phase 1/2: build caches"
  set +e
  printf '%s\n' "${COMBOS[@]}" | xargs -P "$JOBS" -I{} bash -c '
    read -r region init_seed fold <<<"$1"
    echo "[BUILD] start  r=$region is=$init_seed f=$fold"
    PYTHONHASHSEED="$init_seed" python -m core.tstr \
      --model trtr --region "$region" \
      --init_seed "$init_seed" --split_seed "$SPLIT_SEED" \
      --fold "$fold" --n_folds "$N_FOLDS" \
      --n_jobs 1 --force_rebuild --no_summary
  ' _ {}
  build_status=$?
  set -e
  [ $build_status -ne 0 ] && echo "[TRTR] WARNING: $build_status from build phase (some combos may have failed)"
fi

# phase 2: write summary sequentially
echo "[TRTR] phase 2/2: write summary"
for combo in "${COMBOS[@]}"; do
  read -r region init_seed fold <<<"$combo"
  PYTHONHASHSEED="$init_seed" python -m core.tstr \
    --model trtr --region "$region" \
    --init_seed "$init_seed" --split_seed "$SPLIT_SEED" \
    --fold "$fold" --n_folds "$N_FOLDS" \
    --n_jobs 1
done

end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Total elapsed time: $elapsed seconds"
