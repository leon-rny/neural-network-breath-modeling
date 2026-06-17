#!/bin/bash
#SBATCH --job-name=tune_val
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --output=/home/rane10/logs/tune_val.o%A_%a
#SBATCH --error=/home/rane10/logs/tune_val.e%A_%a
#SBATCH --array=0-24%25
set -euo pipefail

source /opt/miniforge/etc/profile.d/conda.sh
conda activate nnbm

# Validate the Optuna-tuned config at FULL fidelity (5 folds x 5 seeds, 500 epochs) and compare to the
# committed CVAE baseline (mouth .798 / nose .664). Best params read LIVE from the study journal
# (best.json can be stale). Usage: MODEL=cvae_part REGION=mouth sbatch experiments/19_tune_validate.sh
#        then  AGGREGATE=1 MODEL=cvae_part REGION=mouth sbatch --array=0 experiments/19_tune_validate.sh
MODEL="${MODEL:?set MODEL=cvae|cvae_part}"
REGION="${REGION:?set REGION=mouth|nose}"
EPOCHS="${EPOCHS:-500}"
SEEDS=(0 1 7 42 123)
FOLDS=(1 2 3 4 5)
N_FOLDS=5
SPLIT_SEED=42
CV_MODE="${CV_MODE:-kfold}"                    # kfold (in-distribution) or loso (cross-subject)
SFX=""; CV_FLAGS="--cv_mode $CV_MODE"
[ "$CV_MODE" = "loso" ] && { SFX="_loso"; CV_FLAGS="--cv_mode loso --part_dropout 0.1"; }

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# pull best params from the Optuna study (live), matching cv_mode's study/log naming
read LD ED PED FB BETA LR ALPHA NC WARMUP BS <<EOF
$(python3 - "$MODEL" "$REGION" "$EPOCHS" "$SFX" <<'PY'
import sys, optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
model, region, epochs, sfx = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
st = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(f'results/tuning/{model}_{region}{sfx}_v4.log'))
s = optuna.load_study(study_name=f'{model}_tstr_{region}{sfx}', storage=st)
p = s.best_params
ped = p.get('part_embed_dim', p.get('embed_dim', 8))
warm = max(1, int(p['warmup_frac'] * epochs))
print(p['latent_dim'], p.get('embed_dim', 8), ped, p['free_bits'], p['beta_max'], p['lr'], p['alpha'], p['n_copies'], warm, p['batch_size'])
PY
)
EOF

echo "[VAL] $MODEL $REGION best: ld=$LD ed=$ED ped=$PED fb=$FB beta=$BETA lr=$LR alpha=$ALPHA n_copies=$NC warmup=$WARMUP bs=$BS"
PART_FLAG=""; [ "$MODEL" = "cvae_part" ] && PART_FLAG="--part_embed_dim $PED"

if [ "${AGGREGATE:-0}" = "1" ]; then
  echo "[VAL] aggregate-only: merging tuned-config TSTR rows into results/summary.csv"
  python -m core.tstr --aggregate --model "$MODEL" $CV_FLAGS \
    --regions "$REGION" --init_seeds 0,1,7,42,123 --folds 1,2,3,4,5 \
    --split_seed "$SPLIT_SEED" --n_folds "$N_FOLDS" \
    --latent_dim "$LD" --embed_dim "$ED" $PART_FLAG --free_bits "$FB" --alpha "$ALPHA" --n_copies "$NC"
  exit 0
fi

IDX=${SLURM_ARRAY_TASK_ID:-${TASK_ID:?set SLURM_ARRAY_TASK_ID or TASK_ID=<0..24>}}
FOLD=${FOLDS[$(( IDX % 5 ))]}
SEED=${SEEDS[$(( IDX / 5 ))]}
echo "[VAL] fold=$FOLD seed=$SEED"

PYTHONHASHSEED="$SEED" python -m core.train --model "$MODEL" --region "$REGION" $CV_FLAGS \
  --init_seed "$SEED" --split_seed "$SPLIT_SEED" --fold "$FOLD" --n_folds "$N_FOLDS" \
  --epochs "$EPOCHS" --batch_size "$BS" --latent_dim "$LD" --embed_dim "$ED" $PART_FLAG \
  --free_bits "$FB" --beta_max "$BETA" --lr "$LR" --beta_warmup_epochs "$WARMUP" \
  --alpha "$ALPHA" --n_copies "$NC"

PYTHONHASHSEED="$SEED" python -m core.tstr --model "$MODEL" --region "$REGION" $CV_FLAGS \
  --init_seed "$SEED" --split_seed "$SPLIT_SEED" --fold "$FOLD" --n_folds "$N_FOLDS" \
  --latent_dim "$LD" --embed_dim "$ED" $PART_FLAG --free_bits "$FB" \
  --alpha "$ALPHA" --n_copies "$NC" --n_jobs 1 --no_summary

# Aggregate after the array: AGGREGATE=1 MODEL=$MODEL REGION=$REGION sbatch --array=0 experiments/19_tune_validate.sh
