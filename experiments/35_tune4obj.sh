#!/bin/bash
#SBATCH --job-name=tune4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --output=/home/rane10/logs/tune4.o%A_%a
#SBATCH --error=/home/rane10/logs/tune4.e%A_%a
set -euo pipefail

source /opt/miniforge/etc/profile.d/conda.sh
conda activate nnbm
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# PHASE 4d — tune cvae_part + tpinn-res for 4 objectives (tstr/loso/tstr+/loso+). One array task = one
# manifest row (see core/gen_tune4_manifest.py). Set MANIFEST=<tsv> and EPOCHS, submit with the row count:
#   MANIFEST=results/tuning4/search.tsv   EPOCHS=200 sbatch --array=0-$((N-1))%200 experiments/35_tune4obj.sh
#   MANIFEST=results/tuning4/validate.tsv EPOCHS=500 sbatch --array=0-$((N-1))%200 experiments/35_tune4obj.sh
# (N = data rows = lines-in-tsv minus the header.) hp_tag namespaces every run -> no summary writes (jsons only).
MANIFEST="${MANIFEST:?set MANIFEST=path to manifest tsv}"
EPOCHS="${EPOCHS:?set EPOCHS}"
SPLIT_SEED=42
IDX=${SLURM_ARRAY_TASK_ID:-${TASK_ID:?set SLURM_ARRAY_TASK_ID or TASK_ID}}

LINE=$(sed -n "$((IDX + 2))p" "$MANIFEST")   # +2: 1-indexed sed + skip header
[ -n "$LINE" ] || { echo "[TUNE4] ERROR: no manifest row at idx $IDX"; exit 1; }
IFS=$'\t' read -r MODEL OBJ MODE CV PD REGION HPTAG LD ED PED FB BM ALPHA NCOP AUGR SEED FOLD NF <<< "$LINE"

PARAMS="results/pinn/params_${REGION}.npy"
PHYSFLAGS=""
if [ "$MODEL" = tpinn ]; then
  [ -f "$PARAMS" ] || { echo "[TUNE4] ERROR: missing $PARAMS"; exit 1; }
  PHYSFLAGS="--phys_residual --phys_prep stdscale"
fi
echo "[TUNE4] $OBJ $MODEL region=$REGION hp=$HPTAG ld=$LD ed=$ED ped=$PED fb=$FB bm=$BM a=$ALPHA n=$NCOP augr=$AUGR cv=$CV pd=$PD seed=$SEED fold=$FOLD nf=$NF ep=$EPOCHS"

if [ "${SKIP_TRAIN:-0}" != "1" ]; then
PYTHONHASHSEED="$SEED" python -m core.train --model "$MODEL" --region "$REGION" --cv_mode "$CV" \
  --init_seed "$SEED" --split_seed "$SPLIT_SEED" --fold "$FOLD" --n_folds "$NF" --epochs "$EPOCHS" \
  --part_dropout "$PD" --latent_dim "$LD" --embed_dim "$ED" --part_embed_dim "$PED" \
  --free_bits "$FB" --beta_max "$BM" --alpha "$ALPHA" --n_copies "$NCOP" --hp_tag "$HPTAG" $PHYSFLAGS
fi

MODEFLAGS="--mode $MODE"
[ "$MODE" = tstr_plus ] && MODEFLAGS="$MODEFLAGS --augmentation_ratio $AUGR"
PYTHONHASHSEED="$SEED" python -m core.tstr --model "$MODEL" --region "$REGION" --cv_mode "$CV" \
  --init_seed "$SEED" --split_seed "$SPLIT_SEED" --fold "$FOLD" --n_folds "$NF" \
  --part_dropout "$PD" --latent_dim "$LD" --embed_dim "$ED" --part_embed_dim "$PED" \
  --free_bits "$FB" --alpha "$ALPHA" --n_copies "$NCOP" --hp_tag "$HPTAG" $PHYSFLAGS \
  $MODEFLAGS --n_jobs 1 --no_summary
