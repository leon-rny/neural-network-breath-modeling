#!/bin/bash
set -euo pipefail

REGIONS=(mouth nose)
INIT_SEEDS=(0 1 7 42 123)
FOLDS=(1 2 3 4 5)
VARIANTS=(conv_baseline conv_slim conv_tiny conv_large_kernel conv_asym conv_asym_no_dropout conv_baseline_dropout mlp mlp_small mlp_tiny transformer)
read -r -a CONDS <<<"${CONDS:-true false}"
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
for variant in "${VARIANTS[@]}"; do
  for region in "${REGIONS[@]}"; do
    for cond in "${CONDS[@]}"; do
      for init_seed in "${INIT_SEEDS[@]}"; do
        for fold in "${FOLDS[@]}"; do
          COMBOS+=("$region $variant $init_seed $fold $cond")
        done
      done
    done
  done
done
echo "[ARCH] ${#COMBOS[@]} combos | $JOBS parallel | 1 thread/job | CONDS=(${CONDS[*]})"

start_time=$(date +%s)

# phase 1: train + eval in parallel, write per-combo JSON, skip summary
if [ "$SKIP_TRAIN" = "1" ]; then
  echo "[ARCH] phase 1/2: train+eval SKIPPED (SKIP_TRAIN=1)"
else
  echo "[ARCH] phase 1/2: train + eval"
  set +e
  printf '%s\n' "${COMBOS[@]}" | xargs -P "$JOBS" -I{} bash -c '
    read -r region variant init_seed fold cond <<<"$1"
    echo "[RUN] start  r=$region v=$variant is=$init_seed f=$fold cp=$cond"
    PYTHONHASHSEED="$init_seed" python -m ablations.cvae_architecture \
      --variant "$variant" --region "$region" \
      --init_seed "$init_seed" --split_seed "$SPLIT_SEED" \
      --fold "$fold" --n_folds "$N_FOLDS" \
      --cond_part "$cond" \
      --n_jobs 1 --skip_existing --no_summary
  ' _ {}
  run_status=$?
  set -e
  [ $run_status -ne 0 ] && echo "[ARCH] WARNING: $run_status from train+eval phase (some combos may have failed)"
fi

# phase 2: aggregate per-combo result JSONs into summary.csv
echo "[ARCH] phase 2/2: aggregate summary"
python -m ablations.cvae_architecture --aggregate \
  --variants "$(IFS=, ; echo "${VARIANTS[*]}")" \
  --regions "$(IFS=, ; echo "${REGIONS[*]}")" \
  --init_seeds "$(IFS=, ; echo "${INIT_SEEDS[*]}")" \
  --folds "$(IFS=, ; echo "${FOLDS[*]}")" \
  --cond_parts "$(IFS=, ; echo "${CONDS[*]}")" \
  --split_seed "$SPLIT_SEED" --n_folds "$N_FOLDS"

end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Total elapsed time: $elapsed seconds"
