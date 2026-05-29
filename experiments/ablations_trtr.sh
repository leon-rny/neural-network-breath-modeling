#!/bin/bash
set -euo pipefail

PIPELINES=(replication shap_fix lgbm_fix tsfresh_fix smote_fix)
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
for pipeline in "${PIPELINES[@]}"; do
  for region in "${REGIONS[@]}"; do
    for init_seed in "${INIT_SEEDS[@]}"; do
      for fold in "${FOLDS[@]}"; do
        COMBOS+=("$region $pipeline $init_seed $fold")
      done
    done
  done
done

# "original protocol" baseline: replication pipeline under a single 80/10/10 stratified split (no folds)
LEGACY_COMBOS=()
for region in "${REGIONS[@]}"; do
  for seed in "${INIT_SEEDS[@]}"; do
    LEGACY_COMBOS+=("$region $seed")
  done
done
echo "[ABLATION] ${#COMBOS[@]} k-fold + ${#LEGACY_COMBOS[@]} legacy combos | $JOBS parallel | 1 thread/job"

start_time=$(date +%s)

# phase 1: build caches in parallel
if [ "$SKIP_BUILD" = "1" ]; then
  echo "[ABLATION] phase 1/2: build SKIPPED (SKIP_BUILD=1)"
else
  echo "[ABLATION] phase 1/2: build k-fold caches"
  set +e
  printf '%s\n' "${COMBOS[@]}" | xargs -P "$JOBS" -I{} bash -c '
    read -r region pipeline init_seed fold <<<"$1"
    echo "[BUILD] start  r=$region p=$pipeline is=$init_seed f=$fold"
    PYTHONHASHSEED="$init_seed" python -m ablations.trtr \
      --region "$region" --pipeline "$pipeline" \
      --init_seed "$init_seed" --split_seed "$SPLIT_SEED" \
      --fold "$fold" --n_folds "$N_FOLDS" \
      --n_jobs 1 --force_rebuild --no_summary
  ' _ {}
  build_status=$?
  set -e
  [ $build_status -ne 0 ] && echo "[ABLATION] WARNING: $build_status from k-fold build phase (some combos may have failed)"

  echo "[ABLATION] phase 1/2: build legacy single-split caches"
  set +e
  printf '%s\n' "${LEGACY_COMBOS[@]}" | xargs -P "$JOBS" -I{} bash -c '
    read -r region seed <<<"$1"
    echo "[BUILD-LEGACY] start  r=$region s=$seed"
    PYTHONHASHSEED="$seed" python -m ablations.trtr \
      --region "$region" --pipeline replication --single_split \
      --init_seed "$seed" --split_seed "$seed" \
      --n_jobs 1 --force_rebuild --no_summary
  ' _ {}
  legacy_build_status=$?
  set -e
  [ $legacy_build_status -ne 0 ] && echo "[ABLATION] WARNING: $legacy_build_status from legacy build phase (some combos may have failed)"
fi

# phase 2: write summary sequentially
echo "[ABLATION] phase 2/2: write summary (k-fold)"
for combo in "${COMBOS[@]}"; do
  read -r region pipeline init_seed fold <<<"$combo"
  PYTHONHASHSEED="$init_seed" python -m ablations.trtr \
    --region "$region" --pipeline "$pipeline" \
    --init_seed "$init_seed" --split_seed "$SPLIT_SEED" \
    --fold "$fold" --n_folds "$N_FOLDS" \
    --n_jobs 1
done

echo "[ABLATION] phase 2/2: write summary (legacy single-split)"
for combo in "${LEGACY_COMBOS[@]}"; do
  read -r region seed <<<"$combo"
  PYTHONHASHSEED="$seed" python -m ablations.trtr \
    --region "$region" --pipeline replication --single_split \
    --init_seed "$seed" --split_seed "$seed" \
    --n_jobs 1
done

end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Total elapsed time: $elapsed seconds"
