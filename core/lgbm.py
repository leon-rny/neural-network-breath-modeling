"""LightGBM on the per-trial combo features (core.combo), tuned inside the training split, and its ensemble with the
raw-signal CNN (core.cnn) trained with physics augmentation.

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
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core import tstr as T
from core.combo import combo_features
from core.data import CLASS_TO_IDX, get_split, load_dataset
from core.provenance import fingerprint
from core.utils import seed_everything


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
