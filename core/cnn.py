"""Raw-signal 1D-CNN with physics-informed augmentation (TSTR+ / LOSO+ with a data-hungry learner) and the exchange
rate of the augmentation in equivalent real data.

Per partition: train the CNN on the real training trials (real-only) and on real + synthetic trials, select the epoch
on the real validation split, test on the real test split. The real-only CNN is the matched baseline of every
augmented one (same seed, split, real subset and epochs).

CNN: input baseline-corrected (first 5 samples) humidity / temperature scaled per channel by the real training SD;
Conv(2-32, k5)-BN-ReLU, Conv(32-64, k5)-BN-ReLU, Conv(64-64, k3)-BN-ReLU, global average + max pooling, dropout 0.3,
linear(128-3); Adam 1e-3, weight decay 1e-4, batch 32, 150 epochs, the epoch with the best validation accuracy is kept.
Augmentation policies (--augmentation_ratio 2 = twice as many synthetic as real training trials):
  physgain  physics re-renderings of the real trials (core.augment.rerender): random sensor gain x/÷1.8 per channel
            + the DHT22 observation (0.1 grid, 99.9 %RH ceiling)
  physwide  the same + a random first-order sensor response change (-2 .. +4 s)
  mixup     same-class mixup of real trials (core.tstr.generate_real_aug), the generic reference
--real_fraction f < 1 trains everything on round(f * n) trials of every class x participant stratum of the training
split (at least 2); validation and test splits stay complete.

Usage: python -m core.cnn --region mouth --cv_mode loso --fold 3 --init_seed 0 [--aug_policy physgain,mixup] [--real_fraction 0.25]
Writes results/cnn_plus/<run_id>_tstr.json and one row per policy to results/summary.csv (model cnn_plus; accuracy =
CNN + augmentation, realonly_accuracy = matched real-only CNN, val_* = validation split, subset = frac<f> for f < 1)
and results/mmd.csv.
Exchange rate: python -m core.cnn --aggregate reads results/cnn_plus/*_tstr.json and writes results/exchange_rate.csv.
Per cell (region x protocol), policy and real fraction f: mean test accuracy of the real-only and the augmented CNN
over folds (per seed, and 'all' = over seeds x folds), the real fraction f_equivalent at which the real-only learning
curve (piecewise linear in log2 f) reaches the augmented accuracy, and multiplier = f_equivalent / f ("the synthetic
data are worth multiplier x the real data"); beyond the measured curve f_equivalent is extrapolated with the end
segment (extrapolated = True).
"""

import argparse
import copy
import glob
import json
import os

import numpy as np
import pandas as pd
import torch
from torch import nn

from core import tstr as T
from core.augment import mmd_report, rerender, stream_rng
from core.data import CLASS_TO_IDX, get_split, load_dataset
from core.provenance import fingerprint
from core.utils import seed_everything

POLICIES = {'physgain': (1.8, 0.0), 'physwide': (1.8, 4.0), 'mixup': None}  # (gain_max, tau_max) of the re-rendering


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Conv1d(2, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
                               nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
                               nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, 3))

    def forward(self, x):
        h = self.f(x)
        return self.head(torch.cat([h.mean(2), h.amax(2)], 1))


def prep(signals: np.ndarray, scale: np.ndarray) -> torch.Tensor:
    """Absolute (n, 2, T) trials -> baseline-corrected channels scaled by the real training SD."""
    s = np.asarray(signals, dtype=float)
    s = s - s[:, :, :5].mean(2, keepdims=True)
    return torch.tensor(s / scale[None, :, None], dtype=torch.float32)


class CNNClassifier:
    """predict / predict_proba on absolute signals (for core.tstr.evaluate_classifier)."""

    def __init__(self, net: Net, scale: np.ndarray):
        self.net, self.scale = net.eval(), scale

    def predict_proba(self, signals: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return torch.softmax(self.net(prep(signals, self.scale)), 1).numpy()

    def predict(self, signals: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.net(prep(signals, self.scale)).argmax(1).numpy()


def fit(X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, seed: int, scale: np.ndarray, epochs: int = 150) -> CNNClassifier:
    """Train on (X_tr, y_tr) signals and keep the epoch with the best validation accuracy."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    xt, yt = prep(X_tr, scale), torch.tensor(np.asarray(y_tr), dtype=torch.long)
    xv = prep(X_val, scale)
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    best, state = -1.0, None
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(yt), generator=g)
        for i in range(0, len(yt), 32):
            b = perm[i:i + 32]
            if len(b) < 2:
                continue
            loss = nn.functional.cross_entropy(net(xt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            va = float((net(xv).argmax(1).numpy() == y_val).mean())
        if va > best:
            best, state = va, copy.deepcopy(net.state_dict())
    net.load_state_dict(state)
    return CNNClassifier(net, scale)


def subset_train(df_train: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    """Keep round(fraction * n) trials (at least 2) of every class x participant stratum; order preserved."""
    if fraction >= 1.0:
        return df_train
    rng = np.random.RandomState(seed)
    keep = []
    for _, idx in df_train.groupby(['class', 'participant'], sort=True).groups.items():
        idx = np.sort(np.asarray(idx))
        k = min(len(idx), max(2, int(round(fraction * len(idx)))))
        keep.extend(rng.choice(idx, k, replace=False).tolist())
    return df_train.loc[sorted(keep)].reset_index(drop=True)


def run(region: str, cv_mode: str, fold: int, n_folds: int, init_seed: int, split_seed: int, aug_policies: list[str],
        augmentation_ratio: float, epochs: int, dataset_dir: str, real_fraction: float = 1.0) -> list[dict]:
    """CNN real-only and CNN + each augmentation policy on one partition; one result dict per policy (core.tstr layout)."""
    seed_everything(init_seed)
    df = load_dataset(dataset_dir)
    df = df[df['region'] == region].reset_index(drop=True)
    df_tr, df_va, df_te = get_split(df, cv_mode=cv_mode, fold=fold - 1, n_folds=n_folds, split_seed=split_seed)
    df_tr = subset_train(df_tr, real_fraction, init_seed)
    y = lambda d: np.array([CLASS_TO_IDX[c] for c in d['class']])  # noqa: E731
    R_tr, R_va, R_te = T._signals(df_tr), T._signals(df_va), T._signals(df_te)
    y_tr, y_va, y_te = y(df_tr), y(df_va), y(df_te)
    scale = (R_tr - R_tr[:, :, :5].mean(2, keepdims=True)).std(axis=(0, 2)) + 1e-6
    clf_ro = fit(R_tr, y_tr, R_va, y_va, init_seed, scale, epochs)
    metrics_ro, metrics_ro_val = T.evaluate_classifier(clf_ro, R_te, y_te), T.evaluate_classifier(clf_ro, R_va, y_va)
    bc = lambda s: T.preprocess_synth_signals(s, 'baseline')  # noqa: E731
    results = []
    for policy in aug_policies:
        if policy == 'mixup':
            synth, y_synth = T.generate_real_aug(df_tr, int(round(augmentation_ratio * len(y_tr))), init_seed)
            synth, y_synth = np.asarray(synth, dtype=float), np.asarray(y_synth)
        else:
            gain_max, tau_max = POLICIES[policy]
            copies = max(1, int(round(augmentation_ratio)))
            synth = rerender(R_tr, copies, gain_max, tau_max, stream_rng(init_seed, fold, f'{real_fraction}/{policy}'))
            y_synth = np.tile(y_tr, copies)
        clf = fit(np.concatenate([R_tr, synth]), np.r_[y_tr, y_synth], R_va, y_va, init_seed, scale, epochs)
        metrics = T.evaluate_classifier(clf, R_te, y_te)
        mmd = mmd_report(bc(R_tr), y_tr, bc(synth), y_synth, bc(R_te), with_test=(cv_mode == 'kfold'))
        print(f"[CNN+] {region} {cv_mode} fold {fold} seed {init_seed} f {real_fraction:g} {policy} ({len(y_synth)} synthetic): "
              f"acc {metrics['accuracy']:.4f} | real-only {metrics_ro['accuracy']:.4f} | lift {metrics['accuracy'] - metrics_ro['accuracy']:+.4f} | MMD {mmd['mmd']:.4f}")
        results.append({'model': 'cnn_plus', 'region': region, 'channel': 'both', 'cv_mode': cv_mode, 'init_seed': init_seed,
                        'split_seed': split_seed, 'fold': fold, 'real_fraction': real_fraction, 'n_train_real': int(len(y_tr)),
                        'n_synthetic': int(len(y_synth)), 'augmentation_ratio': augmentation_ratio,
                        'aug_source': 'mixup' if policy == 'mixup' else 'physics', 'aug_policy': policy,
                        'metrics': metrics, 'metrics_realonly': metrics_ro,
                        'metrics_val': T.evaluate_classifier(clf, R_va, y_va), 'metrics_realonly_val': metrics_ro_val,
                        'mmd': mmd, 'subset': f'frac{real_fraction:g}' if real_fraction < 1.0 else '',
                        'prep': 'baseline', 'phys_prep': '', 'provenance': fingerprint(dataset_dir)})
    return results


def f_equivalent(acc: float, fractions: np.ndarray, curve: np.ndarray) -> tuple[float, bool]:
    """Real fraction at which the real-only curve (piecewise linear in log2 f) reaches acc; (f, extrapolated)."""
    lf = np.log2(fractions)
    for i in range(len(fractions) - 1):
        lo, hi = curve[i], curve[i + 1]
        if (lo - acc) * (hi - acc) <= 0 and hi != lo:
            return float(2 ** (lf[i] + (acc - lo) / (hi - lo) * (lf[i + 1] - lf[i]))), False
    i, (ref_lf, ref_acc) = ((len(fractions) - 2, (lf[-1], curve[-1])) if acc > curve[-1] else (0, (lf[0], curve[0])))
    slope = (curve[i + 1] - curve[i]) / (lf[i + 1] - lf[i])
    return (float(2 ** (ref_lf + (acc - ref_acc) / slope)) if slope > 0 else float('nan')), True


def aggregate(results_dir: str = 'results') -> pd.DataFrame:
    """Exchange-rate table from the per-run results (results/cnn_plus/*_tstr.json) -> results/exchange_rate.csv."""
    rows = []
    for path in glob.glob(f'{results_dir}/cnn_plus/*_tstr.json'):
        r = json.load(open(path))
        rows.append({'region': r['region'], 'cv_mode': r['cv_mode'], 'aug_policy': r['aug_policy'], 'augmentation_ratio': r['augmentation_ratio'],
                     'real_fraction': r.get('real_fraction', 1.0), 'seed': r['init_seed'], 'fold': r['fold'],
                     'realonly': 100 * r['metrics_realonly']['accuracy'], 'aug': 100 * r['metrics']['accuracy']})
    runs = pd.DataFrame(rows)
    out = []
    for (region, cv, policy, ratio), g in runs.groupby(['region', 'cv_mode', 'aug_policy', 'augmentation_ratio']):
        for seed, gs in [('all', g)] + list(g.groupby('seed')):
            m = gs.groupby('real_fraction')[['realonly', 'aug']].mean().sort_index()
            n = gs.groupby('real_fraction').size()
            fr = m.index.values.astype(float)
            for f in fr:
                fe, ext = f_equivalent(m.loc[f, 'aug'], fr, m['realonly'].values) if len(fr) > 1 else (float('nan'), True)
                out.append({'region': region, 'cv_mode': cv, 'aug_policy': policy, 'augmentation_ratio': ratio, 'seed': seed,
                            'real_fraction': f, 'n_runs': int(n.loc[f]), 'realonly_accuracy': m.loc[f, 'realonly'],
                            'accuracy': m.loc[f, 'aug'], 'lift': m.loc[f, 'aug'] - m.loc[f, 'realonly'],
                            'f_equivalent': fe, 'multiplier': fe / f, 'extrapolated': ext})
    table = pd.DataFrame(out)
    path = f'{results_dir}/exchange_rate.csv'
    temporary = path + f'.{os.getpid()}.tmp'
    table.to_csv(temporary, index=False)
    os.replace(temporary, path)
    print(f'[EXCHANGE] {len(runs)} runs -> {path} ({len(table)} rows)')
    return table


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--region', choices=['mouth', 'nose'])
    ap.add_argument('--cv_mode', default='kfold', choices=['kfold', 'loso'])
    ap.add_argument('--fold', type=int, help='1-indexed fold (LOSO: held-out participant)')
    ap.add_argument('--n_folds', type=int, default=None, help='default 5 (k-fold) / 8 (LOSO)')
    ap.add_argument('--init_seed', type=int, default=0)
    ap.add_argument('--split_seed', type=int, default=42)
    ap.add_argument('--aug_policy', default='physgain', help=f'comma list of {sorted(POLICIES)}')
    ap.add_argument('--augmentation_ratio', type=float, default=2.0)
    ap.add_argument('--real_fraction', type=float, default=1.0)
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--dataset_dir', default='dataset')
    ap.add_argument('--no_summary', action='store_true')
    ap.add_argument('--aggregate', action='store_true', help='only write results/exchange_rate.csv from results/cnn_plus')
    a = ap.parse_args()
    if a.aggregate:
        aggregate()
        return
    if a.region is None or a.fold is None:
        ap.error('--region and --fold are required')
    policies = a.aug_policy.split(',')
    if not set(policies) <= set(POLICIES):
        ap.error(f'unknown aug_policy in {a.aug_policy!r}; choose from {sorted(POLICIES)}')
    n_folds = a.n_folds or (8 if a.cv_mode == 'loso' else 5)
    results = run(a.region, a.cv_mode, a.fold, n_folds, a.init_seed, a.split_seed, policies, a.augmentation_ratio, a.epochs, a.dataset_dir, a.real_fraction)
    loso = '_loso' if a.cv_mode == 'loso' else ''
    frac = f'_frac{a.real_fraction:g}' if a.real_fraction < 1.0 else ''
    for result in results:
        policy = result['aug_policy']
        result['evaluation_config'] = {k: v for k, v in vars(a).items() if k not in ('dataset_dir', 'no_summary', 'aggregate')} | {'n_folds': n_folds, 'aug_policy': policy, 'hp_tag': policy}
        T.save_result(result, 'cnn_plus', f'{a.region}_s{a.init_seed}_f{a.fold}{loso}{frac}_cnn_{policy}_r{a.augmentation_ratio:g}_e{a.epochs}')
        if not a.no_summary:
            T.save_summary(result)


if __name__ == '__main__':
    main()
