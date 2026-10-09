"""Nested leave-one-participant-out evaluation, one task = (region, outer test participant, inner test participant or
outer task, seed).

Outer loop (evaluation): every participant is the test participant once; validation = the next participant, training =
the other 6. Inner loop (selection only, core.report): the 7 non-test participants rotate as inner test participant,
inner validation = the next of the 7, inner training = the remaining 5 (core.data.nested_split). The outer test
participant is not part of any inner task.

Every task fits the candidates of the selection menus on its own split (training participants; the validation
participant only for the CNN epoch and the generator checkpoint) and stores accuracy, macro F1 and class probabilities
on its test participant:
  stacker        thesis stacker, real trials only (core.stacker)
  lgbm           LightGBM on combo features, Optuna TPE 20 trials on leave-one-participant-out over the training
                 participants, refit on all of them (core.lgbm)
  cnn            1D-CNN, real trials only (core.cnn)
  cnn_physgain   + 2 physics re-renderings per real training trial: gain x/÷1.8 per channel + DHT22 observation
  cnn_postgain   + 2 x n_train PI-CVAE posterior samples of class-balanced real training parents (core.augment), then the
                 physics gain step; PI-CVAE trained on the task's training participants (core.train)
  ens, ens_*     mean class probabilities of the LightGBM and the respective CNN
Outer tasks also compute fixed comparisons and fidelity (never selected):
  cnn_/ens_jitter    + 2 jittered copies per real training trial (SD 0.05 x training SD per channel)
  cnn_/ens_mixup     + 2 mixups per real training trial with a random trial of the same class, lambda ~ Beta(0.4, 0.4)
  cnn_/ens_cvaegain  as postgain with the CVAE (same parents, latent noise and gains)
  tstr_<gen>_<clf>   LightGBM (the task's tuned hyperparameters), CNN and their ensemble trained only on n_train
                     class-balanced prior samples of the CVAE / PI-CVAE decoded with the null participant token
  fidelity           core.fidelity MMD and morphology of the prior samples and of every augmentation set against the
                     held-out participant
Random streams (core.augment.stream_rng) are keyed by seed and task (outer: '<o+1>', inner: '<o+1>.<i+1>'); the stream
names of the menu candidates are those of the development runs (development/nested), whose results this module
reproduces.

Usage (repository root, after experiments/00_calibration.sh):
  python -m core.nested --region mouth --outer 0 --inner -1 --seed 0      (--inner -1: outer task)
Writes results/nested/<task>.json; generator checkpoints in results/nested/work/<task>/.
"""

import argparse
import json
import os
import time
import warnings

import numpy as np
import torch
from sklearn.metrics import f1_score

from core import cnn, stacker
from core.augment import GAIN_MAX, jitter, mixup, posterior_samples, prior_samples, rerender, sample_parents, stream_rng
from core.data import CLASS_TO_IDX, BreathDataset, load_dataset, nested_split, split_frames
from core.fidelity import mmd_report, morphology_report
from core.lgbm import combo_features, make, tune
from core.provenance import file_digest, fingerprint
from core.train import train
from core.utils import seed_everything

MENU = ('stacker', 'lgbm', 'cnn', 'cnn_physgain', 'cnn_postgain', 'ens', 'ens_physgain', 'ens_postgain')
CONTROLS = ('jitter', 'mixup', 'cvaegain')
GENERATORS = ('cvae', 'picvae')
N_TRIALS = 20  # Optuna trials of the LightGBM tuning
COPIES = 2  # synthetic trials per real training trial


def task_name(region: str, outer: int, inner: int, seed: int) -> str:
    return f'{region}_o{outer}_' + ('outer' if inner < 0 else f'i{inner}') + f'_s{seed}'


def scores(P: np.ndarray, y: np.ndarray) -> dict:
    pred = P.argmax(1)
    return {'accuracy': float((pred == y).mean()), 'macro_f1': float(f1_score(y, pred, average='macro')), 'proba': np.round(P, 6).tolist()}


def run(region: str, outer: int, inner: int, seed: int, out_dir: str = 'results/nested', dataset_dir: str = 'dataset',
        calibration_dir: str = 'results/pinn', gen_epochs: int = 500, cnn_epochs: int = cnn.EPOCHS, n_trials: int = N_TRIALS) -> dict:
    t0 = time.time()
    task = task_name(region, outer, inner, seed)
    work = os.path.join(out_dir, 'work', task)
    os.makedirs(work, exist_ok=True)
    seed_everything(seed)
    df = load_dataset(dataset_dir)
    df = df[df['region'] == region].reset_index(drop=True)
    train_p, val_p, test_p = nested_split(sorted(df['participant'].unique()), outer, inner)
    tr, va, te = split_frames(df, train_p, val_p, test_p)
    key = outer + 1 if inner < 0 else f'{outer + 1}.{inner + 1}'
    y = lambda d: np.array([CLASS_TO_IDX[c] for c in d['class']])  # noqa: E731
    signals = lambda d: np.stack([np.stack([h, t]) for h, t in zip(d['humidity'], d['temperature'])]).astype(float)  # noqa: E731
    y_tr, y_va, y_te = y(tr), y(va), y(te)
    R_tr, R_va, R_te = signals(tr), signals(va), signals(te)
    print(f'[NESTED] {task}: train {train_p} | val {val_p} | test {test_p} ({len(y_te)} trials)', flush=True)
    res, fidelity = {}, {}
    gen = lambda name: train(name, region, tr, va, seed, os.path.join(work, f'{name}.pt'), {'task': task, 'train': train_p, 'val': val_p, 'test': test_p},  # noqa: E731
                             dataset_dir, calibration_dir, gen_epochs)

    # real-only classifiers
    res['stacker'] = scores(stacker.fit_predict(tr, te, seed), y_te)
    X_tr, X_te = combo_features(R_tr).values, combo_features(R_te).values
    params, inner_acc = tune(X_tr, y_tr, tr['participant'].values, seed, n_trials)
    lgbm = make(params, seed).fit(X_tr, y_tr)
    P_lgbm = lgbm.predict_proba(X_te)
    res['lgbm'] = scores(P_lgbm, y_te)

    # CNN real-only / + physics gain / + PI-CVAE posterior samples + gain, and the ensembles with the LightGBM
    scale = (R_tr - R_tr[:, :, :5].mean(2, keepdims=True)).std(axis=(0, 2)) + 1e-6
    n_add = COPIES * len(y_tr)
    synth = {'physgain': (rerender(R_tr, COPIES, GAIN_MAX, 0.0, stream_rng(seed, key, '1.0/physgain')), np.tile(y_tr, COPIES))}
    parents, y_post = sample_parents(tr, n_add, stream_rng(seed, key, 'postgain_parents'))
    gens = {'picvae': gen('picvae')}
    post = posterior_samples(gens['picvae'], parents, stream_rng(seed, key, 'postgain_z'))
    synth['postgain'] = (rerender(post, 1, GAIN_MAX, 0.0, stream_rng(seed, key, 'postgain_gain')), y_post)
    if inner < 0:
        gens['cvae'] = gen('cvae')
        post = posterior_samples(gens['cvae'], parents, stream_rng(seed, key, 'postgain_z'))
        synth['cvaegain'] = (rerender(post, 1, GAIN_MAX, 0.0, stream_rng(seed, key, 'postgain_gain')), y_post)
        synth['jitter'] = (jitter(R_tr, COPIES, BreathDataset(tr).stats['std'], stream_rng(seed, key, 'jitter')), np.tile(y_tr, COPIES))
        synth['mixup'] = (mixup(R_tr, y_tr, COPIES, stream_rng(seed, key, 'mixup')), np.tile(y_tr, COPIES))
    P_cnn = cnn.fit(R_tr, y_tr, R_va, y_va, seed, scale, cnn_epochs).predict_proba(R_te)
    res['cnn'], res['ens'] = scores(P_cnn, y_te), scores((P_lgbm + P_cnn) / 2, y_te)
    for name, (S, y_s) in synth.items():
        P = cnn.fit(np.concatenate([R_tr, S]), np.r_[y_tr, y_s], R_va, y_va, seed, scale, cnn_epochs).predict_proba(R_te)
        res[f'cnn_{name}'], res[f'ens_{name}'] = scores(P, y_te), scores((P_lgbm + P) / 2, y_te)

    # outer task: synthetic-only training and fidelity of the generators' prior samples (null participant token)
    if inner < 0:
        for name, g in gens.items():
            S, y_s = prior_samples(g, len(y_tr), stream_rng(seed, key, 'prior_z'))
            synth[f'{name}_prior'] = (S, y_s)
            P_l = make(params, seed).fit(combo_features(S).values, y_s).predict_proba(X_te)
            P_c = cnn.fit(S, y_s, R_va, y_va, seed, scale, cnn_epochs).predict_proba(R_te)
            res[f'tstr_{name}_lgbm'], res[f'tstr_{name}_cnn'], res[f'tstr_{name}_ens'] = scores(P_l, y_te), scores(P_c, y_te), scores((P_l + P_c) / 2, y_te)
        for name, (S, y_s) in synth.items():
            fidelity[name] = {'n': len(y_s), **mmd_report(S, y_s, R_tr, R_te, y_te), **morphology_report(S, y_s, R_tr, y_tr, R_te, y_te)}

    print(f'[NESTED] {task}: ' + ' '.join(f'{c} {r["accuracy"]:.3f}' for c, r in res.items()) + f' | {time.time() - t0:.0f}s', flush=True)
    provenance = fingerprint(dataset_dir) | {'calibration_sha256': file_digest(f'{calibration_dir}/params_{region}.npy')}
    out = {'config': {'region': region, 'outer': outer, 'inner': inner, 'seed': seed, 'gen_epochs': gen_epochs, 'cnn_epochs': cnn_epochs, 'n_trials': n_trials},
           'task': task, 'train': train_p, 'val': val_p, 'test': test_p, 'y_test': y_te.tolist(), 'n_train': len(y_tr),
           'lgbm_params': params, 'lgbm_inner': inner_acc, 'provenance': provenance, 'results': res, 'fidelity': fidelity,
           'seconds': time.time() - t0}
    with open(os.path.join(out_dir, f'{task}.json.tmp'), 'w') as f:
        json.dump(out, f, default=str)
    os.replace(os.path.join(out_dir, f'{task}.json.tmp'), os.path.join(out_dir, f'{task}.json'))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--region', required=True, choices=['mouth', 'nose'])
    ap.add_argument('--outer', type=int, required=True, help='0-indexed outer test participant (sorted participants)')
    ap.add_argument('--inner', type=int, required=True, help='0-indexed inner test participant among the others; -1 = outer task')
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--out_dir', default='results/nested')
    ap.add_argument('--dataset_dir', default='dataset')
    ap.add_argument('--calibration_dir', default='results/pinn')
    ap.add_argument('--gen_epochs', type=int, default=500, help='generator epochs (smoke tests only)')
    ap.add_argument('--cnn_epochs', type=int, default=cnn.EPOCHS, help='CNN epochs (smoke tests only)')
    ap.add_argument('--n_trials', type=int, default=N_TRIALS, help='LightGBM tuning trials (smoke tests only)')
    a = ap.parse_args()
    warnings.filterwarnings('ignore')
    torch.set_num_threads(1)
    run(a.region, a.outer, a.inner, a.seed, a.out_dir, a.dataset_dir, a.calibration_dir, a.gen_epochs, a.cnn_epochs, a.n_trials)


if __name__ == '__main__':
    main()
