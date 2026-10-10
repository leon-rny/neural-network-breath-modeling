"""Within-participant (k-fold) fidelity, the counterpart of the fidelity of the nested LOSO outer tasks (core.nested).

Thesis protocol per region: class-stratified 5-fold split of the trials (split seed 42, the same folds for every seed),
validation = 15 % of the training folds (class-stratified, generator checkpoint), test = the held-out fold
(core.data.kfold_split). Training and test trials come from the same participants, whereas the LOSO test participant
is unseen. One task = (region, fold, seed): the PI-CVAE and the CVAE are trained on the training trials (core.train)
and the synthetic sets of the LOSO outer tasks (core.nested.synthetic_sets: physgain, postgain, cvaegain, post,
cvaepost, jitter, mixup; prior samples with the null participant token) are compared with the test fold by
core.fidelity (MMD and morphology; reference: the real training trials vs. the test fold). No classifiers.
Random streams are keyed by seed and 'k<fold+1>'.

Usage (repository root, after experiments/00_calibration.sh):
  python -m core.kfold --region mouth --fold 0 --seed 0
Writes results/kfold/<task>.json; generator checkpoints in results/kfold/work/<task>/.
"""

import argparse
import json
import os
import time
import warnings

import torch

from core.augment import prior_samples, stream_rng
from core.data import kfold_split, load_dataset
from core.fidelity import mmd_report, morphology_report
from core.nested import labels, signals, synthetic_sets
from core.provenance import file_digest, fingerprint
from core.train import train
from core.utils import seed_everything

N_FOLDS = 5
SPLIT_SEED = 42


def task_name(region: str, fold: int, seed: int) -> str:
    return f'{region}_k{fold}_s{seed}'


def run(region: str, fold: int, seed: int, out_dir: str = 'results/kfold', dataset_dir: str = 'dataset',
        calibration_dir: str = 'results/pinn', gen_epochs: int = 500) -> dict:
    t0 = time.time()
    task = task_name(region, fold, seed)
    work = os.path.join(out_dir, 'work', task)
    os.makedirs(work, exist_ok=True)
    seed_everything(seed)
    df = load_dataset(dataset_dir)
    tr, va, te = kfold_split(df[df['region'] == region], fold, N_FOLDS, SPLIT_SEED)
    key = f'k{fold + 1}'
    y_tr, y_te = labels(tr), labels(te)
    R_tr, R_te = signals(tr), signals(te)
    print(f'[KFOLD] {task}: train {len(tr)} | val {len(va)} | test {len(te)} trials', flush=True)
    split = {'task': task, 'fold': fold, 'n_folds': N_FOLDS, 'split_seed': SPLIT_SEED}
    gens = {name: train(name, region, tr, va, seed, os.path.join(work, f'{name}.pt'), split, dataset_dir, calibration_dir, gen_epochs)
            for name in ('picvae', 'cvae')}
    synth = synthetic_sets(tr, R_tr, y_tr, gens, seed, key, controls=True)
    for name, g in gens.items():
        synth[f'{name}_prior'] = prior_samples(g, len(y_tr), stream_rng(seed, key, 'prior_z'))
    fidelity = {name: {'n': len(y_s), **mmd_report(S, y_s, R_tr, R_te, y_te), **morphology_report(S, y_s, R_tr, y_tr, R_te, y_te)}
                for name, (S, y_s) in synth.items()}

    print(f'[KFOLD] {task}: MMD ' + ' '.join(f'{n} {f["mmd"]:.4f}' for n, f in fidelity.items())
          + f' | real {fidelity["physgain"]["mmd_real_ref"]:.4f} | {time.time() - t0:.0f}s', flush=True)
    provenance = fingerprint(dataset_dir) | {'calibration_sha256': file_digest(f'{calibration_dir}/params_{region}.npy')}
    out = {'config': {'region': region, 'fold': fold, 'seed': seed, 'n_folds': N_FOLDS, 'split_seed': SPLIT_SEED, 'gen_epochs': gen_epochs},
           'task': task, 'n_train': len(tr), 'n_val': len(va), 'n_test': len(te),
           'test_participants': te['participant'].value_counts().sort_index().to_dict(), 'provenance': provenance,
           'fidelity': fidelity, 'seconds': time.time() - t0}
    with open(os.path.join(out_dir, f'{task}.json.tmp'), 'w') as f:
        json.dump(out, f, default=str)
    os.replace(os.path.join(out_dir, f'{task}.json.tmp'), os.path.join(out_dir, f'{task}.json'))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--region', required=True, choices=['mouth', 'nose'])
    ap.add_argument('--fold', type=int, required=True, help=f'0-indexed test fold (0..{N_FOLDS - 1})')
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--out_dir', default='results/kfold')
    ap.add_argument('--dataset_dir', default='dataset')
    ap.add_argument('--calibration_dir', default='results/pinn')
    ap.add_argument('--gen_epochs', type=int, default=500, help='generator epochs (smoke tests only)')
    a = ap.parse_args()
    warnings.filterwarnings('ignore')
    torch.set_num_threads(1)
    run(a.region, a.fold, a.seed, a.out_dir, a.dataset_dir, a.calibration_dir, a.gen_epochs)


if __name__ == '__main__':
    main()
