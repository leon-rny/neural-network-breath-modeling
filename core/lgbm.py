"""LightGBM on the per-trial combo features, tuned inside the training split, and its ensemble with the
raw-signal CNN (core.cnn) trained with physics augmentation.

Combo features: hand-crafted breathing features + spectral / wavelet features + catch22, computed on the
baseline-corrected trial (first 5 samples subtracted per channel).
  breath    (86) amplitude statistics, rise / fall slopes and their timing, rise times, duty cycle, area, peaks and
                 inter-peak intervals, high-pass oscillation, dominant frequency, spectral centroid / entropy,
                 autocorrelation per channel + humidity–temperature coupling (correlation, lag, range ratio, peak delay)
  spectral  (88) log power and relative FFT power spectrum of the detrended trial, relative Welch band powers
                 (nperseg 16), db2 wavelet level energies / statistics per channel
  catch22   (48) catch22 + mean / SD (catch24) per channel (pycatch22)
Per-trial extraction has no fitted state; imputation, scaling and selection happen inside the training folds.

Classifier: median imputation -> standard scaling -> [top-k features by ANOVA F, k in {20, 50, 100, all}] -> LightGBM.
Hyperparameters (k, n_estimators, num_leaves, learning_rate, min_child_samples, colsample_bytree, reg_lambda) by
Optuna TPE (n_trials, 5 random start-up trials) maximizing the inner cross-validation accuracy inside the training split:
k-fold -> stratified 5-fold; LOSO -> leave-one-participant-out over the training participants. Then refit on the whole
training split and scored on the test split (and the validation split, for information). Trained on real data only.

Ensemble (--ensemble): the mean of the class probabilities of this LightGBM and of the CNN (core.cnn: real training
trials + 2 physics re-renderings each with random sensor gain x/÷1.8 + DHT22 observation, or with --aug_policy postgain
2 PI-CVAE posterior samples per real trial passed through the same gain step; epoch chosen on the real validation
split). Matched reference: the same ensemble with the real-only CNN.

Usage: python -m core.lgbm --region mouth --cv_mode loso --fold 3 --init_seed 0 [--ensemble]
Writes results/<model>/<run_id>_tstr.json and rows to results/summary.csv:
  without --ensemble:
  lgbm            accuracy = LightGBM (real-only)
  with --ensemble (the LightGBM is tuned and fitted identically, its accuracy is kept in the ensemble's json):
  cnn_plus        accuracy = CNN + augmentation, realonly_accuracy = CNN real-only (as core.cnn; + results/mmd.csv)
  lgbm_cnn_plus   accuracy = ensemble with CNN + augmentation, realonly_accuracy = ensemble with the real-only CNN
  (aug_policy = physgain or postgain)
"""

import argparse

import numpy as np
import optuna
import pandas as pd
from scipy import signal as sps
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core import tstr as T
from core.data import CLASS_TO_IDX, get_split, load_dataset
from core.provenance import fingerprint
from core.utils import seed_everything

DT = 2.004  # s per sample
CH = ('H', 'T')


def _breath_channel(x: np.ndarray) -> dict:
    t = np.arange(len(x)) * DT
    rng = x.max() - x.min() + 1e-9
    d = np.diff(x)
    f = {'max': x.max(), 'min': x.min(), 'range': rng, 'mean': x.mean(), 'std': x.std(), 'median': np.median(x),
         'iqr': stats.iqr(x), 'skew': stats.skew(x), 'kurt': stats.kurtosis(x), 'energy': np.mean(x ** 2),
         'mad_diff': np.mean(np.abs(d)), 'std_diff': d.std(), 'max_rise': d.max(), 'max_fall': d.min(),
         't_peak': t[np.argmax(x)], 't_trough': t[np.argmin(x)], 't_max_rise': t[np.argmax(d)], 't_max_fall': t[np.argmin(d)]}
    for q in (0.1, 0.5, 0.9):  # rise time to q x peak (relative to the start level 0)
        above = np.nonzero(x >= q * x.max())[0] if x.max() > 0 else np.array([])
        f[f't_rise{int(q * 100)}'] = t[above[0]] if len(above) else t[-1]
    f['duty50'] = np.mean(x >= 0.5 * x.max()) if x.max() > 0 else 0.0
    f['end_rel'] = x[-1] / rng
    f['auc'] = x.sum() * DT
    f['centroid_t'] = (t * np.abs(x)).sum() / (np.abs(x).sum() + 1e-9)
    hp = x - np.convolve(x, np.ones(5) / 5, mode='same')  # moving-average high-pass
    peaks, props = sps.find_peaks(x, prominence=0.05 * rng)
    f['n_peaks'] = len(peaks)
    f['peak_prom_mean'] = props['prominences'].mean() if len(peaks) else 0.0
    ipi = np.diff(peaks) * DT
    f['ipi_mean'] = ipi.mean() if len(ipi) else 0.0
    f['ipi_std'] = ipi.std() if len(ipi) > 1 else 0.0
    f['hp_std'] = hp.std()
    f['hp_zero_cross'] = np.sum(np.diff(np.sign(hp - hp.mean())) != 0)
    spec = np.abs(np.fft.rfft(x - np.linspace(x[0], x[-1], len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), DT)
    p = spec[1:] / (spec[1:].sum() + 1e-12)
    f['dom_freq'] = freqs[1:][np.argmax(spec[1:])]
    f['dom_power'] = p.max()
    f['spec_centroid'] = (freqs[1:] * p).sum()
    f['spec_entropy'] = -(p * np.log(p + 1e-12)).sum()
    xc = x - x.mean()
    ac = np.correlate(xc, xc, mode='full')[len(x) - 1:] / (np.dot(xc, xc) + 1e-12)
    for lag in (1, 2, 3, 5, 8):
        f[f'ac{lag}'] = ac[lag]
    zc = np.nonzero(ac < 0)[0]
    f['ac_first_zero'] = zc[0] if len(zc) else len(x)
    return f


def breath_features(S: np.ndarray) -> pd.DataFrame:
    rows = []
    for s in S:
        f = {}
        for c, x in zip(CH, s):
            f.update({f'{c}_{k}': v for k, v in _breath_channel(x).items()})
        h, tt = s
        f['HT_corr'] = np.corrcoef(h, tt)[0, 1] if h.std() > 0 and tt.std() > 0 else 0.0
        cc = np.correlate(h - h.mean(), tt - tt.mean(), mode='full')
        f['HT_lag'] = (np.argmax(cc) - (len(h) - 1)) * DT
        f['HT_range_ratio'] = (h.max() - h.min()) / (tt.max() - tt.min() + 1e-9)
        f['HT_peak_dt'] = (np.argmax(h) - np.argmax(tt)) * DT
        rows.append(f)
    return pd.DataFrame(rows)


def spectral_features(S: np.ndarray) -> pd.DataFrame:
    import pywt
    rows = []
    level = pywt.dwt_max_level(S.shape[2], 'db2')
    for s in S:
        f = {}
        for c, x in zip(CH, s):
            xd = x - np.linspace(x[0], x[-1], len(x))
            mag = np.abs(np.fft.rfft(xd))
            tot = (mag ** 2).sum() + 1e-12
            f[f'{c}_logpower'] = np.log(tot)
            f.update({f'{c}_fft{i}': m ** 2 / tot for i, m in enumerate(mag[1:], start=1)})
            _, pw = sps.welch(x - x.mean(), fs=1 / DT, nperseg=16)
            pw = pw / (pw.sum() + 1e-12)
            f.update({f'{c}_welch{i}': v for i, v in enumerate(pw)})
            for li, coef in enumerate(pywt.wavedec(x, 'db2', level=level)):
                f[f'{c}_wl{li}_energy'] = np.sum(coef ** 2)
                f[f'{c}_wl{li}_absmean'] = np.mean(np.abs(coef))
                f[f'{c}_wl{li}_std'] = coef.std()
                f[f'{c}_wl{li}_absmax'] = np.max(np.abs(coef))
        rows.append(f)
    return pd.DataFrame(rows)


def catch22_features(S: np.ndarray) -> pd.DataFrame:
    import pycatch22
    rows = []
    for s in S:
        f = {}
        for c, x in zip(CH, s):
            r = pycatch22.catch22_all(list(map(float, x)), catch24=True)
            f.update({f'{c}_{n}': v for n, v in zip(r['names'], r['values'])})
        rows.append(f)
    return pd.DataFrame(rows)


def combo_features(signals: np.ndarray) -> pd.DataFrame:
    """(n, 222) combo features of (n, 2, T) absolute trials (baseline-corrected here); +/-inf -> NaN (imputed in fold)."""
    S = np.asarray(signals, dtype=float)
    S = S - S[:, :, :5].mean(2, keepdims=True)
    X = pd.concat([breath_features(S).add_prefix('br_'), spectral_features(S).add_prefix('sp_'), catch22_features(S).add_prefix('ca_')], axis=1)
    return X.replace([np.inf, -np.inf], np.nan)


def make(params: dict, seed: int) -> Pipeline:
    from lightgbm import LGBMClassifier
    p = dict(params)
    k = p.pop('k', 'all')
    steps = [('imp', SimpleImputer(strategy='median')), ('scale', StandardScaler())]
    if k != 'all':
        steps.append(('sel', SelectKBest(f_classif, k=int(k))))
    steps.append(('clf', LGBMClassifier(n_jobs=1, random_state=seed, verbose=-1, subsample=0.8, subsample_freq=1, **p)))
    return Pipeline(steps)


def suggest(trial) -> dict:
    return {'k': trial.suggest_categorical('k', [20, 50, 100, 'all']),
            'n_estimators': trial.suggest_int('n_estimators', 100, 600, step=100),
            'num_leaves': trial.suggest_int('num_leaves', 7, 63),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 40),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 10.0, log=True)}


def tune(X: np.ndarray, y: np.ndarray, groups: np.ndarray, cv_mode: str, seed: int, n_trials: int) -> tuple[dict, float]:
    """Best hyperparameters and their mean inner-CV accuracy (training split only)."""
    if cv_mode == 'loso':
        inner = list(LeaveOneGroupOut().split(X, y, groups))
    else:
        inner = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y))

    def objective(trial):
        params = suggest(trial)
        return float(np.mean([(make(params, seed).fit(X[a], y[a]).predict(X[b]) == y[b]).mean() for a, b in inner]))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=5))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, float(study.best_value)


class Probabilities:
    """Fixed class probabilities with the classifier interface of core.tstr.evaluate_classifier."""

    def __init__(self, P: np.ndarray):
        self.P = P

    def predict_proba(self, _X):
        return self.P

    def predict(self, _X):
        return self.P.argmax(1)


def run(region: str, cv_mode: str, fold: int, n_folds: int, init_seed: int, split_seed: int, n_trials: int, ensemble: bool,
        aug_policy: str, augmentation_ratio: float, epochs: int, dataset_dir: str) -> list[dict]:
    seed_everything(init_seed)
    df = load_dataset(dataset_dir)
    df = df[df['region'] == region].reset_index(drop=True)
    df_tr, df_va, df_te = get_split(df, cv_mode=cv_mode, fold=fold - 1, n_folds=n_folds, split_seed=split_seed)
    y = lambda d: np.array([CLASS_TO_IDX[c] for c in d['class']])  # noqa: E731
    y_tr, y_va, y_te = y(df_tr), y(df_va), y(df_te)
    X_tr, X_va, X_te = (combo_features(T._signals(d)).values for d in (df_tr, df_va, df_te))
    params, inner = tune(X_tr, y_tr, df_tr['participant'].values, cv_mode, init_seed, n_trials)
    clf = make(params, init_seed).fit(X_tr, y_tr)
    print(f"[LGBM] {region} {cv_mode} fold {fold} seed {init_seed}: inner {inner:.4f} | test {(clf.predict(X_te) == y_te).mean():.4f} | {params}", flush=True)
    base = {'region': region, 'channel': 'both', 'cv_mode': cv_mode, 'init_seed': init_seed, 'split_seed': split_seed, 'fold': fold,
            'n_train_real': int(len(y_tr)), 'prep': 'baseline', 'phys_prep': '', 'provenance': fingerprint(dataset_dir)}
    results = [{**base, 'model': 'lgbm', 'aug_source': 'none', 'params': params, 'inner_accuracy': inner, 'n_trials': n_trials,
                'metrics': T.evaluate_classifier(clf, X_te, y_te), 'metrics_val': T.evaluate_classifier(clf, X_va, y_va)}]
    if not ensemble:
        return results
    from core import cnn
    cnn_results, cnns = cnn.run(region, cv_mode, fold, n_folds, init_seed, split_seed, [aug_policy], augmentation_ratio, epochs, dataset_dir, return_models=True)
    R_va, R_te = T._signals(df_va), T._signals(df_te)
    P_te, P_va = clf.predict_proba(X_te), clf.predict_proba(X_va)
    ens = lambda m, R, P: Probabilities((P + m.predict_proba(R)) / 2)  # noqa: E731
    lgbm_metrics = results[0]['metrics']
    results = cnn_results  # the lgbm row comes from the plain run (07_lightgbm.sh); here: cnn_plus + the ensemble
    generator = {k: cnn_results[0][k] for k in ('generator', *cnn.GENERATOR_FIELDS, 'phys_variant') if k in cnn_results[0]}
    results.append({**base, **generator, 'model': 'lgbm_cnn_plus', 'metrics_lgbm': lgbm_metrics, 'aug_source': cnn_results[0]['aug_source'], 'aug_policy': aug_policy, 'augmentation_ratio': augmentation_ratio,
                    'n_synthetic': cnn_results[0]['n_synthetic'], 'params': params,
                    'metrics': T.evaluate_classifier(ens(cnns[aug_policy], R_te, P_te), None, y_te),
                    'metrics_realonly': T.evaluate_classifier(ens(cnns['real'], R_te, P_te), None, y_te),
                    'metrics_val': T.evaluate_classifier(ens(cnns[aug_policy], R_va, P_va), None, y_va),
                    'metrics_realonly_val': T.evaluate_classifier(ens(cnns['real'], R_va, P_va), None, y_va)})
    m = results[-1]
    print(f"[LGBM+CNN] {region} {cv_mode} fold {fold} seed {init_seed}: ensemble {m['metrics']['accuracy']:.4f} | with real-only CNN {m['metrics_realonly']['accuracy']:.4f}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--region', required=True, choices=['mouth', 'nose'])
    ap.add_argument('--cv_mode', default='kfold', choices=['kfold', 'loso'])
    ap.add_argument('--fold', type=int, required=True, help='1-indexed fold (LOSO: held-out participant)')
    ap.add_argument('--n_folds', type=int, default=None, help='default 5 (k-fold) / 8 (LOSO)')
    ap.add_argument('--init_seed', type=int, default=0)
    ap.add_argument('--split_seed', type=int, default=42)
    ap.add_argument('--n_trials', type=int, default=20, help='Optuna TPE trials of the inner-CV tuning')
    ap.add_argument('--ensemble', action='store_true', help='also the CNN (core.cnn) and the soft-voting ensemble')
    ap.add_argument('--aug_policy', default='physgain', help='CNN augmentation of the ensemble (core.cnn policies, e.g. physgain, postgain)')
    ap.add_argument('--augmentation_ratio', type=float, default=2.0)
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--dataset_dir', default='dataset')
    ap.add_argument('--no_summary', action='store_true')
    a = ap.parse_args()
    n_folds = a.n_folds or (8 if a.cv_mode == 'loso' else 5)
    results = run(a.region, a.cv_mode, a.fold, n_folds, a.init_seed, a.split_seed, a.n_trials, a.ensemble, a.aug_policy,
                  a.augmentation_ratio, a.epochs, a.dataset_dir)
    loso = '_loso' if a.cv_mode == 'loso' else ''
    for r in results:
        if r['model'] == 'cnn_plus':  # same tag and file name as core.cnn
            tag, run_id = a.aug_policy, f'{a.region}_s{a.init_seed}_f{a.fold}{loso}_cnn_{a.aug_policy}_r{a.augmentation_ratio:g}_e{a.epochs}'
        else:
            tag = f'combo_tpe{a.n_trials}' + (f'_{a.aug_policy}_r{a.augmentation_ratio:g}_e{a.epochs}' if r['model'] == 'lgbm_cnn_plus' else '')
            run_id = f'{a.region}_s{a.init_seed}_f{a.fold}{loso}_{r["model"]}_{tag}'
        r['evaluation_config'] = {k: v for k, v in vars(a).items() if k not in ('dataset_dir', 'no_summary')} | {'n_folds': n_folds, 'hp_tag': tag}
        T.save_result(r, r['model'], run_id)
        if not a.no_summary:
            T.save_summary(r)


if __name__ == '__main__':
    main()
