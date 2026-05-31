#!/bin/bash
#SBATCH --job-name=arch_beta
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem-per-cpu=2G
#SBATCH --time=24:00:00
#SBATCH --output=/home/rane10/logs/arch_beta.o%j
#SBATCH --error=/home/rane10/logs/arch_beta.e%j
set -euo pipefail

source /opt/miniforge/etc/profile.d/conda.sh
conda activate nnbm

VARIANTS=(conv_baseline conv_large_kernel mlp conv_slim)
BETA_MAXES=(0.01 0.1 0.5)
REGIONS=(mouth nose)
INIT_SEEDS=(0 1 7 42 123)
FOLDS=(1 2 3 4 5)
SPLIT_SEED=42
N_FOLDS=5
LATENT_DIM=16
CONFIG=joint
export SPLIT_SEED N_FOLDS
SKIP_TRAIN="${SKIP_TRAIN:-0}"

# parallelism
JOBS="${JOBS:-${SLURM_CPUS_PER_TASK:-50}}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# pre-flight: TRTR cache
echo "[ARCH-BETA] checking TRTR caches ..."
missing=0
for region in "${REGIONS[@]}"; do
  for init_seed in "${INIT_SEEDS[@]}"; do
    for fold in "${FOLDS[@]}"; do
      cache="results/trtr/${region}_is${init_seed}_ss${SPLIT_SEED}_fold${fold}of${N_FOLDS}_checkpoint.pkl"
      if [ ! -f "$cache" ]; then
        echo "  [MISSING CACHE] $cache"
        missing=$((missing + 1))
      fi
    done
  done
done
if [ "$missing" -ne 0 ] && [ "$SKIP_TRAIN" != "1" ]; then
  echo "[ARCH-BETA] ABORT: $missing TRTR cache(s) missing. Build them sequentially first"
  echo "            (e.g. run a single combo per region/init/fold) to avoid a parallel rebuild race."
  exit 1
fi

# build combos
COMBOS=()
for variant in "${VARIANTS[@]}"; do
  for beta_max in "${BETA_MAXES[@]}"; do
    for region in "${REGIONS[@]}"; do
      for init_seed in "${INIT_SEEDS[@]}"; do
        for fold in "${FOLDS[@]}"; do
          COMBOS+=("$variant $beta_max $region $init_seed $fold")
        done
      done
    done
  done
done
echo "[ARCH-BETA] ${#COMBOS[@]} combos | $JOBS parallel | 1 thread/job"

start_time=$(date +%s)

# phase 1: train + eval in parallel
if [ "$SKIP_TRAIN" = "1" ]; then
  echo "[ARCH-BETA] phase 1/2: train+eval SKIPPED (SKIP_TRAIN=1)"
else
  echo "[ARCH-BETA] phase 1/2: train + eval"
  set +e
  printf '%s\n' "${COMBOS[@]}" | xargs -P "$JOBS" -I{} bash -c '
    read -r variant beta_max region init_seed fold <<<"$1"
    echo "[RUN] start  v=$variant b=$beta_max r=$region is=$init_seed f=$fold"
    PYTHONHASHSEED="$init_seed" python -m ablations.cvae_training_dynamics \
      --config "'"$CONFIG"'" --variant "$variant" --beta_max "$beta_max" \
      --region "$region" --init_seed "$init_seed" --split_seed "'"$SPLIT_SEED"'" \
      --fold "$fold" --n_folds "'"$N_FOLDS"'" \
      --latent_dim "'"$LATENT_DIM"'" \
      --n_jobs 1 --skip_existing --no_summary
  ' _ {}
  run_status=$?
  set -e
  [ $run_status -ne 0 ] && echo "[ARCH-BETA] WARNING: $run_status from train+eval phase (some combos may have failed)"
fi

# phase 2: aggregate per-combo JSONs into summary.csv
echo "[ARCH-BETA] phase 2/2: aggregate summary"
python -m ablations.cvae_training_dynamics --aggregate \
  --configs "$CONFIG" \
  --variants "$(IFS=, ; echo "${VARIANTS[*]}")" \
  --beta_maxes "$(IFS=, ; echo "${BETA_MAXES[*]}")" \
  --regions "$(IFS=, ; echo "${REGIONS[*]}")" \
  --init_seeds "$(IFS=, ; echo "${INIT_SEEDS[*]}")" \
  --folds "$(IFS=, ; echo "${FOLDS[*]}")" \
  --split_seed "$SPLIT_SEED" --n_folds "$N_FOLDS"

end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "[ARCH-BETA] Total elapsed time: $elapsed seconds"
