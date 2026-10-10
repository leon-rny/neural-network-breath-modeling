"""One task of the evaluation: nested LOSO (classifiers, augmentation, controls, TSTR, fidelity) or k-fold (fidelity).

Usage: python -m core.nested --region mouth --seed 0 --outer 0 --inner -1   (nested LOSO; --inner -1 = outer task)
       python -m core.nested --region mouth --seed 0 --fold 0               (k-fold)
Writes results/nested/<task>.json or results/kfold/<task>.json, generator checkpoints in <out_dir>/work/<task>/.
"""

import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from core.augment import (
    GAIN_MAX,
    jitter,
    mixup,
    posterior_samples,
    prior_samples,
    rerender,
    sample_parents,
    stream_rng,
)
from core.data import (
    CLASS_TO_IDX,
    BreathDataset,
    kfold_split,
    load_dataset,
    nested_split,
    split_frames,
)
from core.fidelity import mmd_report, morphology_report
from core.train import train
from core.utils import file_digest, fingerprint, seed_everything
from models import cnn, stacker
from models.lgbm import combo_features, make, tune

# selection menu (core.report): real-only, + physics gain, + PI-CVAE posterior samples + gain; ens = LightGBM + CNN
MENU = ('stacker', 'lgbm', 'cnn', 'cnn_physgain', 'cnn_postgain', 'ens', 'ens_physgain', 'ens_postgain')
N_TRIALS = 20  # Optuna trials of the LightGBM tuning
COPIES = 2  # synthetic trials per real training trial
N_FOLDS = 5
SPLIT_SEED = 42  # k-fold split (thesis protocol)

def task_name(region: str, outer: int, inner: int, seed: int) -> str:
    return f'{region}_o{outer}_' + ('outer' if inner < 0 else f'i{inner}') + f'_s{seed}'

def kfold_task_name(region: str, fold: int, seed: int) -> str:
    return f'{region}_k{fold}_s{seed}'

def scores(P: np.ndarray, y: np.ndarray) -> dict:
    pred = P.argmax(1)
    return {'accuracy': float((pred == y).mean()), 'macro_f1': float(f1_score(y, pred, average='macro')), 'proba': np.round(P, 6).tolist()}

def labels(df: pd.DataFrame) -> np.ndarray:
    return np.array([CLASS_TO_IDX[c] for c in df['class']])

def signals(df: pd.DataFrame) -> np.ndarray:
    return np.stack([np.stack([h, t]) for h, t in zip(df['humidity'], df['temperature'])]).astype(float)

def synthetic_sets(tr: pd.DataFrame, R_tr: np.ndarray, y_tr: np.ndarray, gens: dict, seed: int, key: int | str, controls: bool) -> dict:
    """Synthetic training sets of one split, COPIES x n_train trials each.

    :param tr: training trials
    :param R_tr: their absolute signals (n, 2, T)
    :param y_tr: their labels
    :param gens: trained generators {'picvae': ..., 'cvae': ... (controls only)}
    :param key: task key of the random streams
    :param controls: also build cvaegain, post, cvaepost, jitter and mixup
    :return: {name: (absolute signals, labels)}, physgain and postgain always
    """
    # stream names as in the development runs, whose results this reproduces
    n_add = COPIES * len(y_tr)
    sets = {'physgain': (rerender(R_tr, COPIES, GAIN_MAX, 0.0, stream_rng(seed, key, '1.0/physgain')), np.tile(y_tr, COPIES))}
    parents, y_post = sample_parents(tr, n_add, stream_rng(seed, key, 'postgain_parents'))
    post = {name: posterior_samples(g, parents, stream_rng(seed, key, 'postgain_z')) for name, g in gens.items()}
    sets['postgain'] = (rerender(post['picvae'], 1, GAIN_MAX, 0.0, stream_rng(seed, key, 'postgain_gain')), y_post)
    if controls:
        sets['cvaegain'] = (rerender(post['cvae'], 1, GAIN_MAX, 0.0, stream_rng(seed, key, 'postgain_gain')), y_post)
        sets['post'], sets['cvaepost'] = (post['picvae'], y_post), (post['cvae'], y_post)
        sets['jitter'] = (jitter(R_tr, COPIES, BreathDataset(tr).stats['std'], stream_rng(seed, key, 'jitter')), np.tile(y_tr, COPIES))
        sets['mixup'] = (mixup(R_tr, y_tr, COPIES, stream_rng(seed, key, 'mixup')), np.tile(y_tr, COPIES))
    return sets

def fidelity_of(synth: dict, R_tr: np.ndarray, y_tr: np.ndarray, R_te: np.ndarray, y_te: np.ndarray) -> dict:
    return {name: {'n': len(y_s), **mmd_report(S, y_s, R_tr, R_te, y_te), **morphology_report(S, y_s, R_tr, y_tr, R_te, y_te)}
            for name, (S, y_s) in synth.items()}

def save(out: dict, out_dir: str, task: str) -> None:
    # write to a temporary file first: an interrupted task never leaves a partial result
    with open(os.path.join(out_dir, f'{task}.json.tmp'), 'w') as f:
        json.dump(out, f, default=str)
    os.replace(os.path.join(out_dir, f'{task}.json.tmp'), os.path.join(out_dir, f'{task}.json'))

def run(region: str, outer: int, inner: int, seed: int, out_dir: str = 'results/nested', dataset_dir: str = 'dataset', calibration_dir: str = 'results/pinn', gen_epochs: int = 500, cnn_epochs: int = cnn.EPOCHS, n_trials: int = N_TRIALS) -> dict:
    """Nested LOSO task: every menu candidate on the task's test participant; outer tasks also the controls, TSTR
    (training on prior samples only) and fidelity. Inner tasks only feed the selection (core.report).

    :param outer: 0-indexed outer test participant
    :param inner: 0-indexed inner test participant among the other 7, -1 = outer task
    :param gen_epochs: generator epochs (lower only for smoke tests)
    :return: the result written to <out_dir>/<task>.json
    """
    # split: validation participant only for the CNN epoch and the generator checkpoint
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
    y_tr, y_va, y_te = labels(tr), labels(va), labels(te)
    R_tr, R_va, R_te = signals(tr), signals(va), signals(te)
    print(f'[NESTED] {task}: train {train_p} | val {val_p} | test {test_p} ({len(y_te)} trials)', flush=True)
    res = {}
    # real-only classifiers
    res['stacker'] = scores(stacker.fit_predict(tr, te, seed), y_te)
    X_tr, X_te = combo_features(R_tr).values, combo_features(R_te).values
    params, inner_acc = tune(X_tr, y_tr, tr['participant'].values, seed, n_trials)
    lgbm = make(params, seed).fit(X_tr, y_tr)
    P_lgbm = lgbm.predict_proba(X_te)
    res['lgbm'] = scores(P_lgbm, y_te)
    # CNN on real-only and every synthetic set, ensembles with the LightGBM (the CNN reseeds per fit, every set has its
    # own random streams, so the order of the fits does not matter)
    scale = (R_tr - R_tr[:, :, :5].mean(2, keepdims=True)).std(axis=(0, 2)) + 1e-6
    split = {'task': task, 'train': train_p, 'val': val_p, 'test': test_p}
    gens = {name: train(name, region, tr, va, seed, os.path.join(work, f'{name}.pt'), split, dataset_dir, calibration_dir, gen_epochs)
            for name in (('picvae', 'cvae') if inner < 0 else ('picvae',))}
    synth = synthetic_sets(tr, R_tr, y_tr, gens, seed, key, controls=inner < 0)
    P_cnn = cnn.fit(R_tr, y_tr, R_va, y_va, seed, scale, cnn_epochs).predict_proba(R_te)
    res['cnn'], res['ens'] = scores(P_cnn, y_te), scores((P_lgbm + P_cnn) / 2, y_te)
    for name, (S, y_s) in synth.items():
        P = cnn.fit(np.concatenate([R_tr, S]), np.r_[y_tr, y_s], R_va, y_va, seed, scale, cnn_epochs).predict_proba(R_te)
        res[f'cnn_{name}'], res[f'ens_{name}'] = scores(P, y_te), scores((P_lgbm + P) / 2, y_te)
    # outer task: TSTR on prior samples (null participant token) and fidelity
    fidelity = {}
    if inner < 0:
        for name, g in gens.items():
            S, y_s = prior_samples(g, len(y_tr), stream_rng(seed, key, 'prior_z'))
            synth[f'{name}_prior'] = (S, y_s)
            P_l = make(params, seed).fit(combo_features(S).values, y_s).predict_proba(X_te)
            P_c = cnn.fit(S, y_s, R_va, y_va, seed, scale, cnn_epochs).predict_proba(R_te)
            res[f'tstr_{name}_lgbm'], res[f'tstr_{name}_cnn'], res[f'tstr_{name}_ens'] = scores(P_l, y_te), scores(P_c, y_te), scores((P_l + P_c) / 2, y_te)
        fidelity = fidelity_of(synth, R_tr, y_tr, R_te, y_te)
    # save
    print(f'[NESTED] {task}: ' + ' '.join(f'{c} {r["accuracy"]:.3f}' for c, r in res.items()) + f' | {time.time() - t0:.0f}s', flush=True)
    provenance = fingerprint(dataset_dir) | {'calibration_sha256': file_digest(f'{calibration_dir}/params_{region}.npy')}
    out = {'config': {'region': region, 'outer': outer, 'inner': inner, 'seed': seed, 'gen_epochs': gen_epochs, 'cnn_epochs': cnn_epochs, 'n_trials': n_trials},
           'task': task, 'train': train_p, 'val': val_p, 'test': test_p, 'y_test': y_te.tolist(), 'n_train': len(y_tr),
           'lgbm_params': params, 'lgbm_inner': inner_acc, 'provenance': provenance, 'results': res, 'fidelity': fidelity,
           'seconds': time.time() - t0}
    save(out, out_dir, task)
    return out

def run_kfold(region: str, fold: int, seed: int, out_dir: str = 'results/kfold', dataset_dir: str = 'dataset', calibration_dir: str = 'results/pinn', gen_epochs: int = 500) -> dict:
    """k-fold task (thesis protocol, training and test trials from the same participants): fidelity of the outer-task
    synthetic sets and of the prior samples against the test fold, no classifiers.

    :param fold: 0-indexed test fold of the class-stratified N_FOLDS split
    :param gen_epochs: generator epochs (lower only for smoke tests)
    :return: the result written to <out_dir>/<task>.json
    """
    # split: validation = 15 % of the training folds, for the generator checkpoint
    t0 = time.time()
    task = kfold_task_name(region, fold, seed)
    work = os.path.join(out_dir, 'work', task)
    os.makedirs(work, exist_ok=True)
    seed_everything(seed)
    df = load_dataset(dataset_dir)
    tr, va, te = kfold_split(df[df['region'] == region], fold, N_FOLDS, SPLIT_SEED)
    key = f'k{fold + 1}'
    y_tr, y_te = labels(tr), labels(te)
    R_tr, R_te = signals(tr), signals(te)
    print(f'[KFOLD] {task}: train {len(tr)} | val {len(va)} | test {len(te)} trials', flush=True)
    # generators, synthetic sets and fidelity
    split = {'task': task, 'fold': fold, 'n_folds': N_FOLDS, 'split_seed': SPLIT_SEED}
    gens = {name: train(name, region, tr, va, seed, os.path.join(work, f'{name}.pt'), split, dataset_dir, calibration_dir, gen_epochs)
            for name in ('picvae', 'cvae')}
    synth = synthetic_sets(tr, R_tr, y_tr, gens, seed, key, controls=True)
    for name, g in gens.items():
        synth[f'{name}_prior'] = prior_samples(g, len(y_tr), stream_rng(seed, key, 'prior_z'))
    fidelity = fidelity_of(synth, R_tr, y_tr, R_te, y_te)
    # save
    print(f'[KFOLD] {task}: MMD ' + ' '.join(f'{n} {f["mmd"]:.4f}' for n, f in fidelity.items())
          + f' | real {fidelity["physgain"]["mmd_real_ref"]:.4f} | {time.time() - t0:.0f}s', flush=True)
    provenance = fingerprint(dataset_dir) | {'calibration_sha256': file_digest(f'{calibration_dir}/params_{region}.npy')}
    out = {'config': {'region': region, 'fold': fold, 'seed': seed, 'n_folds': N_FOLDS, 'split_seed': SPLIT_SEED, 'gen_epochs': gen_epochs},
           'task': task, 'n_train': len(tr), 'n_val': len(va), 'n_test': len(te),
           'test_participants': te['participant'].value_counts().sort_index().to_dict(), 'provenance': provenance,
           'fidelity': fidelity, 'seconds': time.time() - t0}
    save(out, out_dir, task)
    return out

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # task: --outer and --inner (nested LOSO) or --fold (k-fold)
    p.add_argument('--region', required=True, choices=['mouth', 'nose'])
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--outer', type=int)
    p.add_argument('--inner', type=int)
    p.add_argument('--fold', type=int)
    # paths (out_dir default: results/nested or results/kfold)
    p.add_argument('--out_dir')
    p.add_argument('--dataset_dir', default='dataset')
    p.add_argument('--calibration_dir', default='results/pinn')
    # smoke tests only
    p.add_argument('--gen_epochs', type=int, default=500)
    p.add_argument('--cnn_epochs', type=int, default=cnn.EPOCHS)
    p.add_argument('--n_trials', type=int, default=N_TRIALS)
    a = p.parse_args()
    if (a.outer is not None and a.inner is not None) == (a.fold is not None):
        p.error('give either --outer and --inner (nested LOSO) or --fold (k-fold)')
    return a

def main() -> None:
    a = parse_args()
    warnings.filterwarnings('ignore')
    torch.set_num_threads(1)
    if a.fold is not None:
        run_kfold(a.region, a.fold, a.seed, a.out_dir or 'results/kfold', a.dataset_dir, a.calibration_dir, a.gen_epochs)
    else:
        run(a.region, a.outer, a.inner, a.seed, a.out_dir or 'results/nested', a.dataset_dir, a.calibration_dir, a.gen_epochs, a.cnn_epochs, a.n_trials)

if __name__ == '__main__':
    main()
