#!/bin/bash
set -euo pipefail

REGIONS=(mouth nose)
INIT_SEEDS=(0 1 7 42 123)
FOLDS=(1 2 3 4 5)
CONFIGS=(
  baseline
  a0.025_n5
  a0.025_n10
  a0.05_n5
  a0.05_n10
  a0.1_n5
  a0.1_n10
)
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
for config in "${CONFIGS[@]}"; do
  for region in "${REGIONS[@]}"; do
    for init_seed in "${INIT_SEEDS[@]}"; do
      for fold in "${FOLDS[@]}"; do
        COMBOS+=("$region $config $init_seed $fold")
      done
    done
  done
done
echo "[JITTER] ${#COMBOS[@]} combos | $JOBS parallel | 1 thread/job"

start_time=$(date +%s)

# phase 1: train + eval in parallel, write per-combo JSON, skip summary
if [ "$SKIP_TRAIN" = "1" ]; then
  echo "[JITTER] phase 1/2: train+eval SKIPPED (SKIP_TRAIN=1)"
else
  echo "[JITTER] phase 1/2: train + eval"
  set +e
  printf '%s\n' "${COMBOS[@]}" | xargs -P "$JOBS" -I{} bash -c '
    read -r region config init_seed fold <<<"$1"
    echo "[RUN] start  r=$region c=$config is=$init_seed f=$fold"
    PYTHONHASHSEED="$init_seed" python -m ablations.cvae_jittering \
      --config "$config" --region "$region" \
      --init_seed "$init_seed" --split_seed "$SPLIT_SEED" \
      --fold "$fold" --n_folds "$N_FOLDS" \
      --n_jobs 1 --skip_existing --no_summary
  ' _ {}
  run_status=$?
  set -e
  [ $run_status -ne 0 ] && echo "[JITTER] WARNING: $run_status from train+eval phase (some combos may have failed)"
fi

# phase 2: aggregate per-combo result JSONs into summary.csv
echo "[JITTER] phase 2/2: aggregate summary"
python -m ablations.cvae_jittering --aggregate \
  --configs "$(IFS=, ; echo "${CONFIGS[*]}")" \
  --regions "$(IFS=, ; echo "${REGIONS[*]}")" \
  --init_seeds "$(IFS=, ; echo "${INIT_SEEDS[*]}")" \
  --folds "$(IFS=, ; echo "${FOLDS[*]}")" \
  --split_seed "$SPLIT_SEED" --n_folds "$N_FOLDS"

end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Total elapsed time: $elapsed seconds"
