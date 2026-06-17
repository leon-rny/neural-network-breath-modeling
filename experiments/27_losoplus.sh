#!/bin/bash
#SBATCH --job-name=losoplus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=06:00:00
#SBATCH --output=/home/rane10/logs/losoplus.o%A
#SBATCH --error=/home/rane10/logs/losoplus.e%A
set -euo pipefail

source /opt/miniforge/etc/profile.d/conda.sh
conda activate nnbm
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# LOSO+ : does synthetic augmentation help CROSS-SUBJECT generalization? For each held-out subject,
# train the committed cvae_part under LOSO (part_dropout 0.1 -> learned null token), generate synthetic
# from the null token (= an unseen subject), mix into the real n-1-subject training set, and score the
# held-out subject. Compare to trtr real-only LOSO (mouth ceiling ~0.782, CVAE-TSTR ~0.666).
# Single sequential task (subjects x ratios) to avoid summary.csv write races.
REGION=mouth
COMMON="--model cvae_part --region $REGION --cv_mode loso --part_dropout 0.1 --init_seed 0 --split_seed 42 \
  --n_folds 5 --latent_dim 16 --embed_dim 8 --part_embed_dim 8 --free_bits 0.0 --alpha 0.05 --n_copies 10"

# guard: data must have exactly 5 LOSO subjects
N_DATA=$(python -c "from core.data import load_dataset, n_loso_folds; print(n_loso_folds(load_dataset('dataset')))")
[ "$N_DATA" = "5" ] || { echo "[LOSO+] ERROR: $N_DATA subjects != 5"; exit 1; }

for F in 1 2 3 4 5; do
  PYTHONHASHSEED=0 python -m core.train $COMMON --fold "$F" --epochs 500 --beta_max 0.01
  for R in 0.5 1.0 2.0; do
    echo "==LOSOPLUS_RESULT fold=$F ratio=$R=="
    PYTHONHASHSEED=0 python -m core.tstr $COMMON --fold "$F" --mode tstr_plus --augmentation_ratio "$R" --n_jobs 1 2>&1 \
      | grep -iE "test accuracy|augment" | tail -2
  done
done
echo "==LOSOPLUS_DONE=="
