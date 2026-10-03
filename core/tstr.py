import argparse
import json
import os
import pickle
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import shap
import torch
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.neighbors import NearestNeighbors
from tsfresh import extract_features
from tsfresh.feature_extraction.settings import from_columns
from tsfresh.utilities.dataframe_functions import (
    impute_dataframe_range,
)
from xgboost import XGBClassifier

from core.augment import POLICY_KINDS, augment, mmd_report, top_features
from core.data import (
    CLASS_TO_IDX,
    CLASSES,
    BreathDataset,
    cir_marker,
    get_split,
    load_dataset,
    loso_path_tag,
    phys_prep_marker,
    prep_marker,
    sensor_marker,
    subset_tag,
)
from core.features import TrainingFeatureSelector
from core.provenance import artifact_lock, atomic_json, digest, fingerprint
from core.utils import seed_everything
from models.diffusion import ConditionalDiffusion
from models.gan import Generator as GANGenerator
from models.pinn import PhysicsInformedCVAE, RhythmTPINN, SharedTransportPINN
from models.vae import CVAE, VAE, ClasswiseVAE


def _cv_marker(cv_mode: str) -> str:
    """Return the run-id suffix for a non-kfold cv mode (empty for kfold)."""
    return '' if cv_mode == 'kfold' else f'_{cv_mode}'

def _drop_marker(part_dropout: float) -> str:
    """Return the run-id suffix for a nonzero participant dropout rate."""
    return '' if part_dropout == 0.0 else f'_drop{part_dropout}'

# cli
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser()
    # general
    p.add_argument('--model', required=True, choices=['trtr', 'vae', 'cvae', 'cvae_part', 'pinn', 'tpinn', 'rhythm', 'diffusion', 'gan'])
    p.add_argument('--region', choices=['mouth', 'nose'], default=None)
    p.add_argument('--mode', choices=['tstr', 'tstr_plus'], default='tstr')
    p.add_argument('--ensemble_model', type=str, default='')  # tstr: mix synthetic from a 2nd generator (e.g. tpinn) for a two-generator ensemble
    p.add_argument('--aug_source', choices=['gen', 'mixup', 'jitter', 'smote'], default='gen')  # tstr_plus non-generator controls: gen=generator synthetic; mixup=same-class convex combo; jitter=§4.4.1 Gaussian jitter of real trials; smote=signal-space k-NN SMOTE
    p.add_argument('--channel', choices=['humidity', 'temperature', 'both'], default='both')
    p.add_argument('--n_synthetic', type=int, default=None)
    p.add_argument('--augmentation_ratio', type=float, default=1.0)
    p.add_argument('--aug_policy', choices=['prior', *POLICY_KINDS], default='prior')  # tstr_plus physics policies (core/augment.py): prior samples, boundary / typical-feature steering, rejection-filtered prior samples
    p.add_argument('--aug_stream', default='')  # named random stream of the policy ('' = tp0 / fp / rh15 by generator, as in the development runs)
    p.add_argument('--real_fraction', type=float, default=1.0)  # tstr_plus: stratified fraction of real training data (data-scarcity curve)
    p.add_argument('--dataset_dir', default='dataset')
    p.add_argument('--n_jobs', type=int, default=4)
    p.add_argument('--force_rebuild', action='store_true')
    p.add_argument('--no_summary', action='store_true')
    p.add_argument('--init_seed', type=int, default=42)  # varies across experiments to characterize sensitivity
    p.add_argument('--split_seed', type=int, default=42)  # always fixed!
    p.add_argument('--fold', type=int, default=1)
    p.add_argument('--n_folds', type=int, default=5)
    p.add_argument('--cv_mode', choices=['kfold', 'loso'], default='kfold')
    p.add_argument('--part_dropout', type=float, default=0.0)
    p.add_argument('--loso_trial_val', action='store_true')
    p.add_argument('--loso_exclude', default='')
    # model-specific
    p.add_argument('--free_bits', type=float, default=0.0)
    p.add_argument('--classwise_vae', action='store_true')
    p.add_argument('--latent_dim', type=int, default=16)
    p.add_argument('--embed_dim', type=int, default=8)
    p.add_argument('--part_embed_dim', type=int, default=8)
    p.add_argument('--lambda_phys', type=float, default=1.0)
    p.add_argument('--tau_s', type=float, default=15.0)
    p.add_argument('--temperature_tau', type=float, default=None)
    p.add_argument('--subj_adv_lambda', type=float, default=0.0)
    p.add_argument('--diff_hidden', type=int, default=64)
    p.add_argument('--n_steps', type=int, default=200)
    p.add_argument('--guidance', type=float, default=-1.0)
    p.add_argument('--gan_hidden', type=int, default=64)
    p.add_argument('--gan_loss', choices=['bce', 'hinge'], default='bce')
    p.add_argument('--gan_lr_d', type=float, default=None)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--phys_residual', action='store_true')
    p.add_argument('--ode', action='store_true')  # UDE mode (must match the trained checkpoint)
    p.add_argument('--class_transport', action='store_true')
    p.add_argument('--parametric_source', action='store_true')
    p.add_argument('--learn_cir_params', action='store_true')
    p.add_argument('--hp_tag', default='')
    # jittering augmentation
    p.add_argument('--alpha', type=float, default=0.0)
    p.add_argument('--n_copies', type=int, default=1)
    # validation-split sanity eval
    p.add_argument('--eval_val', action='store_true')
    # aggregate mode
    p.add_argument('--aggregate', action='store_true')
    p.add_argument('--regions', default=None)
    p.add_argument('--init_seeds', default=None)
    p.add_argument('--folds', default=None)
    p.add_argument('--include_subjects', default='')
    p.add_argument('--cir_tag', default='')
    p.add_argument('--preprocessing', choices=['raw', 'baseline', 'peaknorm'], default='raw')
    p.add_argument('--phys_prep', choices=['peakscale', 'shared', 'stdscale'], default='peakscale')
    return p.parse_args()

# utils
def _prep_channel(arr, mode: str = 'raw', nb: int = 5) -> np.ndarray:
    """Apply the selected baseline/peaknorm transform to a single channel array.

    raw=identity, baseline=subtract the per-channel pre-onset baseline (mean of first nb
    samples), peaknorm=baseline then divide by the channel's own peak |amplitude|.
    """
    a = np.asarray(arr, dtype=float)
    if mode in ('', 'raw'):
        return a
    c = a - a[:nb].mean()
    if mode == 'peaknorm':
        pk = np.abs(c).max()
        return c / pk if pk > 1e-8 else c
    return c

def preprocess_signals(df: pd.DataFrame, mode: str = 'raw', nb: int = 5) -> pd.DataFrame:
    """Per-trial signal preprocessing before feature extraction (Phase 4 ablation).

    raw=identity, baseline=subtract per-channel pre-onset baseline (first nb samples),
    peaknorm=baseline then divide each channel by its own peak |amplitude| (removes the
    offset+gain shift). Returns a transformed copy of df.
    """
    if mode in ('', 'raw'):
        return df
    df = df.copy()
    df['humidity'] = df['humidity'].apply(lambda a: _prep_channel(a, mode, nb))
    df['temperature'] = df['temperature'].apply(lambda a: _prep_channel(a, mode, nb))
    return df

def preprocess_synth_signals(signals: np.ndarray, mode: str = 'raw', nb: int = 5) -> np.ndarray:
    """Apply the same per-trial preprocessing to synthetic (n, 2, T) signals, per channel.

    Mirrors preprocess_signals so generated data enters feature extraction in the SAME space
    as the real data (raw=identity, returns the input unchanged).
    """
    if mode in ('', 'raw'):
        return signals
    out = np.asarray(signals, dtype=float).copy()
    for i in range(out.shape[0]):
        out[i, 0, :] = _prep_channel(out[i, 0, :], mode, nb)
        out[i, 1, :] = _prep_channel(out[i, 1, :], mode, nb)
    return out

def df_to_df_long(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Reshape a trial dataframe into the tsfresh long format and class labels.

    :param df: dataframe with per-trial 'humidity'/'temperature' arrays and a 'class' column.
    :return: (df_long, y) where df_long has columns id/time/Humidity/Temperature (one row per
        sample) and y is an int class-index Series indexed by trial id.
    """
    hum_all = np.stack(df['humidity'].values)
    temp_all = np.stack(df['temperature'].values)
    n, T = hum_all.shape
    ids = np.repeat(np.arange(n), T)
    times = np.tile(np.arange(T), n)
    df_long = pd.DataFrame({'id': ids,
                            'time': times,
                            'Humidity': hum_all.ravel(),
                            'Temperature': temp_all.ravel()})
    labels = np.array([CLASS_TO_IDX[c] for c in df['class']])
    y = pd.Series(labels, index=np.arange(n), name='target')
    return df_long, y

def filter_features(features, channel: str):
    """Keep only features whose name starts with the given channel prefix.

    :param features: a feature DataFrame or list of feature names.
    :param channel: 'both' (no filtering), 'humidity', or 'temperature'.
    :return: the same type as features, restricted to the selected channel.
    """
    if channel == 'both':
        return features
    prefix = channel.capitalize() + '__'
    if isinstance(features, pd.DataFrame):
        return features[[c for c in features.columns if c.startswith(prefix)]]
    return [c for c in features if c.startswith(prefix)]

def extract_fixed_features(df_long: pd.DataFrame, top_features_raw: list[str], n_jobs: int = 4, imputation: dict | None = None) -> pd.DataFrame:
    """Extract a fixed set of tsfresh features from long-format data.

    :param df_long: long-format data with id/time/channel columns.
    :param top_features_raw: raw tsfresh feature names to compute (order preserved).
    :param n_jobs: parallel workers for tsfresh.
    :return: DataFrame with exactly top_features_raw as columns, any feature tsfresh fails
        to produce is filled with 0.0.
    """
    kind_to_fc = from_columns(top_features_raw)
    X = extract_features(df_long, column_id='id', column_sort='time', kind_to_fc_parameters=kind_to_fc, n_jobs=n_jobs)
    if imputation is None:
        raise ValueError('Training-fitted imputation values are required; rebuild the real-data cache')
    for col in top_features_raw:
        if col not in X:
            X[col] = np.nan
    impute_dataframe_range(X, imputation['max'], imputation['min'], imputation['median'])
    missing = [c for c in top_features_raw if c not in X.columns]
    if missing:
        for c in missing:
            X[c] = 0.0
    return X[top_features_raw]

class FoldSMOTE(SMOTE):
    """Adapt SMOTE to the training rows available in each inner cross-validation fit."""

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Use random oversampling when a class has only one available training sample."""
        _, counts = np.unique(y, return_counts=True)
        if len(counts) < 2:
            raise ValueError('Classification needs at least two training classes')
        if counts.min() < 2:
            return RandomOverSampler(random_state=self.random_state).fit_resample(X, y)
        sampler = SMOTE(random_state=self.random_state, k_neighbors=min(5, int(counts.min()) - 1))
        return sampler.fit_resample(X, y)

def train_stacking_classifier(X_train: np.ndarray, y_train: np.ndarray, init_seed: int, n_jobs: int = -1, groups: np.ndarray | None = None) -> StackingClassifier:
    """Fit a SMOTE + XGB/CatBoost stacking classifier with a random-forest meta-learner.

    :param X_train: feature matrix.
    :param y_train: integer class labels.
    :param init_seed: random seed for all estimators and SMOTE.
    :param n_jobs: parallel workers for the stacker's cross-validation.
    :param groups: Optional participant groups for disjoint internal stacking folds.
    :return: the fitted StackingClassifier.
    """
    def make_base(clf):
        """Wrap a base estimator in a SMOTE pipeline, or return it unchanged if SMOTE is infeasible."""
        return ImbPipeline([('smote', FoldSMOTE(random_state=init_seed)),
                            ('clf', clf)])

    min_class = int(np.bincount(y_train).min())
    if min_class < 2:
        raise ValueError('Stacking cross-validation needs at least two samples per class')
    n_splits = min(5, min_class)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=init_seed)
    if groups is not None:
        groups = np.asarray(groups)
        if len(groups) != len(y_train) or len(np.unique(groups)) < 2:
            raise ValueError('Grouped stacking needs matching groups and at least two participants')
        cv = list(GroupKFold(n_splits=min(5, len(np.unique(groups)))).split(X_train, y_train, groups))
    # create clf and use smote
    xgb_clf = make_base(XGBClassifier(n_jobs=1, eval_metric='mlogloss', random_state=init_seed, max_depth=4, reg_alpha=0.5, reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8, n_estimators=300))
    cat_clf = make_base(CatBoostClassifier(thread_count=1, logging_level='Silent', random_state=init_seed, iterations=300, depth=4, l2_leaf_reg=5.0, random_strength=2.0, bagging_temperature=2.0, od_type='Iter', od_wait=20, allow_writing_files=False))
    meta_clf = RandomForestClassifier(n_jobs=1, n_estimators=150, max_depth=3, min_samples_leaf=5, min_samples_split=10, random_state=init_seed)
    stacker = StackingClassifier(estimators=[('xgb', xgb_clf), ('cat', cat_clf)], final_estimator=meta_clf, passthrough=True, cv=cv, n_jobs=n_jobs)
    stacker.fit(X_train, y_train)
    return stacker

def evaluate_classifier(clf: StackingClassifier, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Score a fitted classifier on a test set.

    :param clf: a fitted classifier exposing predict/predict_proba.
    :param X_test: test feature matrix.
    :param y_test: integer test labels.
    :return: dict with keys 'accuracy', 'f1_weighted', 'roc_auc_ovr' (nan if it fails),
        'log_loss', and 'per_class_f1' (class name -> f1).
    """
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)
    try:
        roc_auc = float(roc_auc_score(y_test, y_prob, multi_class='ovr'))
    except ValueError as e:
        print(f'ROC-AUC failed: {e}')
        roc_auc = float('nan')
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return {'accuracy': float(accuracy_score(y_test, y_pred)),
            'f1_weighted': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
            'roc_auc_ovr': roc_auc,
            'log_loss': float(log_loss(y_test, y_prob)),
            'per_class_f1': {cls: float(report.get(str(i), {}).get('f1-score', float('nan'))) for i, cls in enumerate(CLASSES)}}

def save_result(result: dict, model: str, run_id: str) -> None:
    """Write a result dict to results/<model>/<run_id>_tstr.json as JSON-safe values.

    :param result: the result dict (nan/numpy values are converted on write).
    :param model: model name, used as the results subdirectory.
    :param run_id: artifact id used as the filename stem.
    :return: None.
    """
    # paths
    os.makedirs(f'results/{model}', exist_ok=True)
    path = f'results/{model}/{run_id}_tstr.json'

    def _json_safe(obj):
        """Recursively convert nan and numpy scalars/arrays into JSON-serializable values."""
        if isinstance(obj, float) and np.isnan(obj):
            return None
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_json_safe(v) for v in obj]
        return obj
    result['artifact_id'] = run_id
    identity = {k: v for k, v in result.items() if k not in ('metrics', 'metrics_val', 'metrics_realonly', 'metrics_realonly_val', 'mmd', 'trtr_metrics', 'feature_overlap', 'top_20_synth_features')}
    result['evaluation_id'] = digest(identity)
    with artifact_lock(path):
        if os.path.exists(path):
            with open(path) as previous:
                if json.load(previous).get('evaluation_id') != result['evaluation_id']:
                    raise ValueError(f'Evaluation identity conflict at {path}; use a new hp_tag or run directory')
        atomic_json(path, _json_safe(result))

def save_summary(result: dict) -> None:
    """Serialize summary updates and deduplicate by complete evaluation identity (results/summary.csv, and
    results/mmd.csv for results that carry a signal-space MMD)."""
    with artifact_lock('results/summary.csv'):
        _save_summary(result)
    if result.get('mmd'):
        with artifact_lock('results/mmd.csv'):
            _save_mmd(result)

def _save_mmd(result: dict) -> None:
    """Append one MMD row (thesis layout: metric, region, model, seed, fold, value, config + run details) to
    results/mmd.csv, keeping the last row per evaluation_id. value = MMD of the synthetic set vs the real
    training trials (core.augment.mmd_report); mmd_class = class-wise mean; mmd_real_ref = real test vs train;
    mmd_test = synthetic set vs the held-out real test trials (k-fold / TSTR+ runs only, floor = mmd_real_ref)."""
    csv_path = 'results/mmd.csv'
    mmd = result['mmd']
    row = pd.DataFrame([{'evaluation_id': result.get('evaluation_id', digest(result)),
                         'metric': 'MMD',
                         'region': result['region'],
                         'model': result['model'],
                         'cv_mode': result.get('cv_mode', 'kfold'),
                         'seed': result.get('init_seed'),
                         'fold': result.get('fold'),
                         'value': mmd['mmd'],
                         'mmd_class': mmd.get('mmd_class'),
                         'mmd_real_ref': mmd.get('mmd_real_ref'),
                         'mmd_test': mmd.get('mmd_test'),
                         'aug_policy': result.get('aug_policy'),
                         'augmentation_ratio': result.get('augmentation_ratio'),
                         'hp_tag': (result.get('evaluation_config') or {}).get('hp_tag'),
                         'config': result.get('artifact_id')}])
    if os.path.exists(csv_path):
        row = pd.concat([pd.read_csv(csv_path), row], ignore_index=True).drop_duplicates(subset=['evaluation_id'], keep='last')
    temporary = csv_path + f'.{os.getpid()}.tmp'
    row.to_csv(temporary, index=False)
    os.replace(temporary, csv_path)

def _save_summary(result: dict) -> None:
    """Append a flattened result row to results/summary.csv, deduping by config keys.

    Reconciles columns with any existing file and keeps the last row per unique
    (model, region, channel, cv_mode, seeds, fold, model/aug hyperparameters, subset,
    phys_variant, cir, prep, phys_prep) combination.

    :param result: a result dict as produced by trtr/tstr/tstr_plus.
    :return: None.
    """
    csv_path = 'results/summary.csv'
    m = result['metrics']
    new_row = {'evaluation_id': result.get('evaluation_id', digest(result)),
               'artifact_id': result.get('artifact_id'),
               'model': result['model'],
               'region': result['region'],
               'channel': result.get('channel', 'both'),
               'cv_mode': result.get('cv_mode', 'kfold'),
               'init_seed': result.get('init_seed'),
               'split_seed': result.get('split_seed'),
               'fold': result.get('fold'),
               'latent_dim': result.get('latent_dim'),
               'embed_dim': result.get('embed_dim'),
               'part_embed_dim': result.get('part_embed_dim'),
               'free_bits': result.get('free_bits'),
               'lambda_phys': result.get('lambda_phys'),
               'alpha': result.get('alpha'),
               'n_copies': result.get('n_copies'),
               'accuracy': m['accuracy'],
               'f1_weighted': m['f1_weighted'],
               'roc_auc_ovr': m['roc_auc_ovr'],
               'log_loss': m['log_loss'],
               'f1_bradypnea': m['per_class_f1']['bradypnea'],
               'f1_eupnea': m['per_class_f1']['eupnea'],
               'f1_tachypnea': m['per_class_f1']['tachypnea'],
               'feature_overlap': result.get('feature_overlap'),
               'n_synthetic': result.get('n_synthetic', result['n_train_real']),
               'n_train_real': result.get('n_train_real'),
               'augmentation_ratio': result.get('augmentation_ratio'),
               'aug_source': result.get('aug_source', 'gen'),
               'aug_policy': result.get('aug_policy'),
               'hp_tag': (result.get('evaluation_config') or {}).get('hp_tag'),
               'realonly_accuracy': (result.get('metrics_realonly') or {}).get('accuracy'),
               'realonly_val_accuracy': (result.get('metrics_realonly_val') or {}).get('accuracy'),
               'mmd': (result.get('mmd') or {}).get('mmd'),
               'mmd_class': (result.get('mmd') or {}).get('mmd_class'),
               'mmd_real_ref': (result.get('mmd') or {}).get('mmd_real_ref'),
               'mmd_test': (result.get('mmd') or {}).get('mmd_test'),
               'subset': result.get('subset', ''),
               'phys_variant': result.get('phys_variant', ''),
               'cir': result.get('cir', ''),
               'prep': result.get('prep', 'raw'),
               'phys_prep': result.get('phys_prep', 'peakscale')}
    # validation-split
    mv = result.get('metrics_val')
    if mv is not None:
        new_row.update({'val_accuracy': mv['accuracy'],
                        'val_f1_weighted': mv['f1_weighted'],
                        'val_roc_auc_ovr': mv['roc_auc_ovr'],
                        'val_log_loss': mv['log_loss'],
                        'val_f1_bradypnea': mv['per_class_f1']['bradypnea'],
                        'val_f1_eupnea': mv['per_class_f1']['eupnea'],
                        'val_f1_tachypnea': mv['per_class_f1']['tachypnea']})
    df_row = pd.DataFrame([new_row])
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        if 'evaluation_id' not in df_old:
            df_old['evaluation_id'] = pd.NA
        for index in df_old.index[df_old['evaluation_id'].isna()]:
            legacy = df_old.loc[index].drop(labels=['evaluation_id']).to_dict()
            df_old.at[index, 'evaluation_id'] = 'legacy_' + digest(legacy)
        if 'cv_mode' not in df_old.columns:
            df_old['cv_mode'] = 'kfold'
        if 'aug_source' not in df_old.columns:
            df_old['aug_source'] = 'gen'  # backfill default so re-runs dedup against legacy rows
        for col in df_row.columns:
            if col not in df_old.columns:
                df_old[col] = pd.NA
        for col in df_old.columns:
            if col not in df_row.columns:
                df_row[col] = pd.NA
            try:
                df_row[col] = df_row[col].astype(df_old[col].dtype)
            except (ValueError, TypeError):
                pass
        combined = pd.concat([df_old, df_row], ignore_index=True)
        # empty-string keys round-trip through CSV as NaN; unify so '' (fresh) and NaN (stored) dedup
        for c in ('subset', 'phys_variant', 'cir'):
            if c in combined.columns:
                combined[c] = combined[c].fillna('')
        df_new = combined.drop_duplicates(subset=['evaluation_id'], keep='last')
    else:
        df_new = df_row
    temporary = csv_path + f'.{os.getpid()}.tmp'
    df_new.to_csv(temporary, index=False)
    os.replace(temporary, csv_path)

# train real test real
def _cache_path(region: str, init_seed: int, split_seed: int, fold: int, n_folds: int, channel: str = 'both', cv_mode: str = 'kfold', loso_tag: str = '', sub_tag: str = '', prep_tag: str = '') -> str:
    """Build the trtr checkpoint cache path for a given configuration."""
    suffix = f'_ch{channel}' if channel != 'both' else ''
    return f'results/trtr/{region}_is{init_seed}_ss{split_seed}_fold{fold}of{n_folds}{_cv_marker(cv_mode)}{loso_tag}{sub_tag}{prep_tag}{suffix}_checkpoint.pkl'

def load_cache(region: str, init_seed: int, split_seed: int, fold: int, n_folds: int, channel: str = 'both', cv_mode: str = 'kfold', loso_tag: str = '', sub_tag: str = '', prep_tag: str = '', dataset_dir: str = 'dataset') -> dict | None:
    """Load the cached trtr checkpoint for a configuration, or None if it does not exist."""
    path = _cache_path(region, init_seed, split_seed, fold, n_folds, channel, cv_mode, loso_tag, sub_tag, prep_tag)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            cache = pickle.load(f)
        if cache.get('provenance') != fingerprint(dataset_dir):
            raise ValueError(f'Stale or unversioned cache: {path}; rebuild in a new run directory')
        return cache
    return None

_SSL_CACHE: dict = {}

def _ssl_embeddings(df: pd.DataFrame, ssl_ckpt: str) -> np.ndarray:
    """SSL contrastive embeddings (research idea #4) for each trial in df, in df row order (aligns with
    df_to_df_long's id order). Signals are baseline-corrected to match the encoder's pretraining space."""
    from core.ssl import SSLEncoder, _baseline_correct, embed
    if ssl_ckpt not in _SSL_CACHE:
        ck = torch.load(ssl_ckpt, map_location='cpu')
        enc = SSLEncoder(ck['emb_dim'])
        enc.load_state_dict(ck['state'])
        enc.eval()
        _SSL_CACHE[ssl_ckpt] = enc
    sigs = np.stack([_baseline_correct(np.stack([r['humidity'], r['temperature']])) for _, r in df.iterrows()])
    return embed(_SSL_CACHE[ssl_ckpt], sigs)

def trtr(dataset_dir: str, region: str, n_jobs: int, init_seed: int, split_seed: int, fold: int, n_folds: int = 5, channel: str = 'both', cv_mode: str = 'kfold', exclude_subjects: tuple = (), loso_trial_val: bool = False, include_subjects: tuple = (), preprocessing: str = 'raw', ssl_ckpt: str = '') -> dict:
    """Run the train-real-test-real pipeline and cache its artifacts.

    Loads/splits the region data, extracts tsfresh features, selects the top-20 by SHAP from a
    tuned LGBM, trains the stacking classifier on real data, evaluates on the real test fold,
    pickles the cache to results/trtr, and returns it.

    :param dataset_dir: dataset root directory.
    :param region: 'mouth' or 'nose'.
    :param n_jobs: parallel workers for feature extraction and the classifier.
    :param init_seed: model/estimator seed.
    :param split_seed: data-split seed.
    :param fold: 1-indexed fold (converted to 0-indexed for get_split).
    :param n_folds: number of cv folds.
    :param channel: 'both', 'humidity', or 'temperature' feature filter.
    :param cv_mode: 'kfold' or 'loso'.
    :param exclude_subjects: 0-indexed subjects to hold out (nested LOSO).
    :param loso_trial_val: use trial-level validation under LOSO.
    :param include_subjects: restrict the pool to these subjects.
    :param preprocessing: 'raw', 'baseline', or 'peaknorm' signal preprocessing.
    :return: cache dict with keys top_20_features_raw, top_20_features_sanitized, X_train_top,
        X_test_top, y_train, y_test, stats, n_train, trtr_metrics, channel, cv_mode, init_seed,
        split_seed, fold, n_folds.
    """
    if ssl_ckpt:
        raise ValueError('SSL evaluation requires a fold-specific pretraining manifest; all-data checkpoints are not valid here')
    # load dataset
    df = load_dataset(dataset_dir)
    df = df[df['region'] == region].reset_index(drop=True)
    df = preprocess_signals(df, preprocessing)
    # fold is 1-indexed
    df_train, _df_val, df_test = get_split(df, cv_mode=cv_mode, fold=fold - 1, n_folds=n_folds, split_seed=split_seed, exclude_subjects=exclude_subjects, loso_trial_val=loso_trial_val, include_subjects=include_subjects)
    stats = BreathDataset(df_train).stats
    ## train real
    df_long_train, y_train = df_to_df_long(df_train)
    # feature extraction
    X_train_raw = extract_features(df_long_train, column_id='id', column_sort='time', n_jobs=n_jobs)
    X_train_raw = filter_features(X_train_raw, channel)
    if X_train_raw.shape[1] == 0:
        raise ValueError(f'No relevant features remain after filtering by channel={channel!r}.')
    # build name mapping for sanitization
    raw_to_san = {col: re.sub(r'[^\w]', '_', col) for col in X_train_raw.columns}
    san_to_raw = {v: k for k, v in raw_to_san.items()}
    X_full_san = X_train_raw.copy()
    X_full_san.columns = [raw_to_san[c] for c in X_full_san.columns]
    # lgbm and shap feature importance
    X_tr, X_val, y_tr, y_val = train_test_split(X_full_san, y_train, test_size=0.2, stratify=y_train, random_state=split_seed)
    param_grid = {'max_depth': [4, 6],
                  'reg_alpha': [0.1, 1.0],
                  'reg_lambda': [0.5, 1.0],
                  'colsample_bytree': [0.8, 1.0]}
    base_lgbm = LGBMClassifier(n_jobs=1, n_estimators=1000, learning_rate=0.05, random_state=init_seed, verbose=-1)
    pipeline = ImbPipeline([('features', TrainingFeatureSelector(n_jobs=n_jobs)), ('model', base_lgbm)])
    grid = {f'model__{key}': value for key, value in param_grid.items()}
    gs = GridSearchCV(pipeline, grid, cv=StratifiedKFold(3, shuffle=True, random_state=split_seed), scoring='accuracy', n_jobs=n_jobs, verbose=0)
    gs.fit(X_tr, y_tr)
    selector = gs.best_estimator_.named_steps['features']
    X_tr, X_val = selector.transform(X_tr), selector.transform(X_val)
    X_full_san = selector.transform(X_full_san)
    imputation = {key: {san_to_raw[col]: value for col, value in values.items()}
                  for key, values in (('max', selector.max_), ('min', selector.min_), ('median', selector.median_))}
    best_params = {key.removeprefix('model__'): value for key, value in gs.best_params_.items()}
    best_lgbm = LGBMClassifier(**best_params, n_jobs=1, n_estimators=1000, learning_rate=0.05, random_state=init_seed, verbose=-1)
    best_lgbm.fit(X_tr.values, y_tr.values, eval_set=[(X_val.values, y_val.values)], eval_metric='multi_logloss', callbacks=[early_stopping(50, verbose=False), log_evaluation(0)])
    explainer = shap.TreeExplainer(best_lgbm)
    shap_values = explainer.shap_values(X_full_san.values)
    mean_abs = np.abs(shap_values).mean(axis=0).mean(axis=1)
    top_idx = np.argsort(mean_abs)[::-1][:20]
    top_20_san = X_full_san.columns.to_numpy()[top_idx].tolist()
    top_20_raw = [san_to_raw[s] for s in top_20_san]
    X_train_top = X_full_san[top_20_san].values
    if ssl_ckpt:
        X_train_top = np.hstack([X_train_top, _ssl_embeddings(df_train, ssl_ckpt)])
    ## test real
    df_long_test, y_test = df_to_df_long(df_test)
    X_test_raw_top = extract_fixed_features(df_long_test, top_20_raw, n_jobs, imputation)
    X_test_san = X_test_raw_top.copy()
    X_test_san.columns = [re.sub(r'[^\w]', '_', col) for col in X_test_san.columns]
    X_test_top = X_test_san[top_20_san].values
    if ssl_ckpt:
        X_test_top = np.hstack([X_test_top, _ssl_embeddings(df_test, ssl_ckpt)])
    # train stack classifier
    clf = train_stacking_classifier(X_train_top, y_train.values, init_seed, n_jobs=n_jobs)
    # evaluate classifier
    trtr_metrics = evaluate_classifier(clf, X_test_top, y_test.values)
    # build cache
    cache = {'provenance': fingerprint(dataset_dir),
             'imputation': imputation,
             'top_20_features_raw': top_20_raw,
             'top_20_features_sanitized': top_20_san,
             'X_train_top': X_train_top,
             'X_test_top': X_test_top,
             'y_train': y_train.values,
             'y_test': y_test.values,
             'stats': stats,
             'n_train': len(df_train),
             'trtr_metrics': trtr_metrics,
             'channel': channel,
             'cv_mode': cv_mode,
             'init_seed': init_seed,
             'split_seed': split_seed,
             'fold': fold,
             'n_folds': n_folds}
    os.makedirs('results/trtr', exist_ok=True)
    with open(_cache_path(region, init_seed, split_seed, fold, n_folds, channel, cv_mode, loso_path_tag(loso_trial_val, exclude_subjects), subset_tag(include_subjects), prep_marker(preprocessing) + ('_ssl' if ssl_ckpt else '')), 'wb') as f:
        pickle.dump(cache, f)
    return cache

# train synthetic test real
def load_model(model_name: str, run_id: str, device: 'torch.device') -> tuple[torch.nn.Module, dict]:
    """Load a trained generator checkpoint and rebuild the matching model.

    :param model_name: one of vae, cvae, cvae_part, pinn, tpinn, diffusion, gan.
    :param run_id: artifact id locating results/<model_name>/<run_id>_checkpoint.pt.
    :param device: torch device to load weights onto.
    :return: (model, stats) where model is the eval-mode network and stats is the
        normalization stats dict saved with the checkpoint.
    """
    # load model
    ckpt_path = f'results/{model_name}/{run_id}_checkpoint.pt'
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # init model
    if model_name == 'vae':
        model = (ClasswiseVAE(ckpt['latent_dim']) if ckpt.get('classwise_vae') else VAE(ckpt['latent_dim']))
    elif model_name == 'cvae':
        model = CVAE(latent_dim=ckpt['latent_dim'], embed_dim=ckpt['embed_dim'])
    elif model_name == 'cvae_part':
        model = CVAE(latent_dim=ckpt['latent_dim'], embed_dim=ckpt['embed_dim'], condition_on_participant=True,
                     num_participants=ckpt.get('num_participants', 3), part_embed_dim=ckpt['part_embed_dim'],
                     part_dropout=ckpt.get('part_dropout', 0.0), subj_adv=ckpt.get('subj_adv', False))
    elif model_name == 'pinn':
        model = PhysicsInformedCVAE(cir_params_init=ckpt['cir_params'],
                                    t_grid=ckpt['t_grid'],
                                    tau_s=ckpt.get('tau_s', 15.0),
                                    learn_cir_params=ckpt.get('learn_cir_params', True),
                                    latent_dim=ckpt['latent_dim'],
                                    num_classes=3,
                                    embed_dim=ckpt['embed_dim'],
                                    condition_on_participant=ckpt.get('condition_on_participant', True),
                                    num_participants=ckpt.get('num_participants', 3),
                                    part_embed_dim=ckpt.get('part_embed_dim', 8),
                                    part_dropout=ckpt.get('part_dropout', 0.0))
    elif model_name == 'tpinn':
        model = SharedTransportPINN(cir_params_init=ckpt['cir_params'],
                                    t_grid=ckpt['t_grid'],
                                    tau_s=ckpt.get('tau_s', 15.0),
                                    learn_transport=ckpt.get('learn_cir_params', False),
                                    residual=ckpt.get('phys_residual', False),
                                    class_transport=ckpt.get('class_transport', False),
                                    parametric_source=ckpt.get('parametric_source', False),
                                    ode=ckpt.get('ode', False),
                                    temperature_tau=ckpt.get('temperature_tau'),
                                    latent_dim=ckpt['latent_dim'],
                                    num_classes=3,
                                    embed_dim=ckpt['embed_dim'],
                                    condition_on_participant=ckpt.get('condition_on_participant', True),
                                    num_participants=ckpt.get('num_participants', 3),
                                    part_embed_dim=ckpt.get('part_embed_dim', 8),
                                    part_dropout=ckpt.get('part_dropout', 0.0))
    elif model_name == 'rhythm':
        model = RhythmTPINN(cir_params_init=ckpt['cir_params'],
                            t_grid=ckpt['t_grid'],
                            tau_s=ckpt.get('tau_s', 15.0),
                            n_nuis=ckpt['latent_dim'],
                            tau_max=ckpt.get('tau_max', 15.0),
                            num_classes=3,
                            embed_dim=ckpt['embed_dim'],
                            condition_on_participant=ckpt.get('condition_on_participant', True),
                            num_participants=ckpt.get('num_participants', 3),
                            part_embed_dim=ckpt.get('part_embed_dim', 8),
                            part_dropout=ckpt.get('part_dropout', 0.0))
    elif model_name == 'diffusion':
        model = ConditionalDiffusion(num_classes=3,
                                     num_participants=ckpt.get('num_participants', 3),
                                     embed_dim=ckpt['embed_dim'],
                                     part_embed_dim=ckpt.get('part_embed_dim', 8),
                                     condition_on_participant=ckpt.get('condition_on_participant', True),
                                     part_dropout=ckpt.get('part_dropout', 0.0),
                                     n_steps=ckpt.get('n_steps', 200),
                                     hidden=ckpt.get('hidden', 64))
    elif model_name == 'gan':
        model = GANGenerator(z_dim=ckpt.get('z_dim', ckpt.get('latent_dim', 16)),
                             num_classes=3,
                             num_participants=ckpt.get('num_participants', 3),
                             embed_dim=ckpt['embed_dim'],
                             part_embed_dim=ckpt.get('part_embed_dim', 8),
                             condition_on_participant=ckpt.get('condition_on_participant', True),
                             part_dropout=ckpt.get('part_dropout', 0.0),
                             hidden=ckpt.get('hidden', 64))
    else:
        raise ValueError(f'Unknown model: {model_name}')
    model.load_state_dict(ckpt['model_state'])
    model.trained_participants = ckpt.get('trained_participants', list(range(getattr(model, 'num_participants', 0))))
    model.provenance = ckpt.get('provenance', {})
    model.to(device).eval()
    return model, ckpt['stats']

def generate_synthetic_signals(model, model_name: str, n_synthetic: int, stats: dict, device: 'torch.device', init_seed: int, participant_idx: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Sample synthetic signals from a generator and denormalize to physical units.

    Splits n_synthetic as evenly as possible across the classes, samples per model family,
    and undoes normalization (physics models use the phys_prep scaling stored in stats).

    :param model: a loaded generator with a .sample method.
    :param model_name: one of vae, cvae, cvae_part, pinn, tpinn, diffusion, gan.
    :param n_synthetic: total number of signals to generate.
    :param stats: normalization stats (mean/std and, for physics models, phys_prep scales).
    :param device: torch device.
    :param init_seed: seed set before sampling for determinism.
    :param participant_idx: participant id to condition on, or None (e.g. LOSO null token).
    :return: (signals, labels) where signals is (n_synthetic, 2, T) in physical units and
        labels is the int class-index array.
    """
    # determine num per class
    if n_synthetic < len(CLASSES):
        raise ValueError('n_synthetic must cover every class')
    n_classes = len(CLASSES)
    n_per_class = n_synthetic // n_classes
    remainder = n_synthetic % n_classes
    counts = [n_per_class + (1 if i < remainder else 0) for i in range(n_classes)]
    mean_t = torch.tensor(stats['mean'], dtype=torch.float32).view(1, 2, 1).to(device)
    std_t = torch.tensor(stats['std'], dtype=torch.float32).view(1, 2, 1).to(device)
    # deterministic pre-sampling seed
    torch.manual_seed(init_seed)
    # generate synthetic signals
    if model_name == 'vae' and not isinstance(model, ClasswiseVAE):
        raise ValueError('An unconditional VAE cannot supply class labels; train with --classwise_vae for TSTR')
    if model_name in ('vae', 'cvae', 'cvae_part', 'diffusion', 'gan'):
        all_signals, all_labels = [], []
        for cls_idx, count in enumerate(counts):
            y_cls = torch.tensor(cls_idx, dtype=torch.long)
            with torch.no_grad():
                z_cls = model.sample(count, y_cls, device, participant=participant_idx)
            all_signals.append((z_cls * std_t + mean_t).cpu().numpy())
            all_labels.append(np.full(count, cls_idx))
        signals_phys = np.concatenate(all_signals, axis=0)
        labels = np.concatenate(all_labels, axis=0)
    elif model_name in ('pinn', 'tpinn', 'rhythm'):
        mode = stats.get('phys_prep', 'peakscale')
        t_mean = float(stats['mean'][1])
        t_std = float(stats['std'][1])
        if mode == 'shared':
            h_dn = t_dn = float(stats['shared_scale'])
            use_tscale = True
        elif mode == 'stdscale':
            h_dn = float(stats['h_bcstd'])
            t_dn = float(stats['t_bcstd'])
            use_tscale = True
        else:  # peakscale (default)
            h_dn = float(stats['h_scale'])
            use_tscale = 't_scale' in stats
            t_dn = float(stats['t_scale']) if use_tscale else None
        all_signals, all_labels = [], []
        for cls_idx, count in enumerate(counts):
            y_cls = torch.tensor(cls_idx, dtype=torch.long)
            with torch.no_grad():
                z_cls = model.sample(count, y_cls, device, participant=participant_idx).cpu()
            h = z_cls[:, 0:1, :] * h_dn
            t = z_cls[:, 1:2, :] * t_dn if use_tscale else z_cls[:, 1:2, :] * t_std + t_mean
            all_signals.append(torch.cat([h, t], dim=1).numpy())
            all_labels.append(np.full(count, cls_idx))
        signals_phys = np.concatenate(all_signals, axis=0)
        labels = np.concatenate(all_labels, axis=0)
    return signals_phys, labels

def validate_generator(model: torch.nn.Module, cache: dict, model_name: str, preprocessing: str) -> None:
    """Reject incompatible data/code, untrained null tokens and mismatched signal spaces."""
    provenance = getattr(model, 'provenance', {})
    if any(provenance.get(key) != value for key, value in cache['provenance'].items()):
        raise ValueError('Generator and real-data cache must use the same versioned data and scientific code')
    settings = provenance['training_config']
    for key in ('init_seed', 'split_seed', 'fold', 'n_folds', 'cv_mode'):
        if settings[key] != cache[key]:
            raise ValueError(f'Generator/cache split mismatch: {key}')
    if model_name in ('pinn', 'tpinn', 'rhythm') and preprocessing == 'raw':
        raise ValueError('Physics generators require baseline or peaknorm classifier preprocessing')
    if cache['cv_mode'] == 'loso' and getattr(model, '_cond_part', False) and model.part_dropout <= 0:
        raise ValueError('LOSO generation requires a trained null participant token')

def tstr(cache: dict, model_name: str, region: str, n_synthetic: int, n_jobs: int, device, init_seed: int, split_seed: int, fold: int, run_id: str, free_bits: float = 0.0, latent_dim: int = 16, embed_dim: int = 8, part_embed_dim: int = 8, lambda_phys: float = 0.0, alpha: float = 0.0, n_copies: int = 1, eval_val: bool = False, dataset_dir: str = 'dataset', n_folds: int = 5, cv_mode: str = 'kfold', guidance: float = -1.0, include_subjects: tuple = (), preprocessing: str = 'raw', ckpt_run_id: str | None = None, ensemble_model: str = '', ensemble_ckpt_run_id: str | None = None) -> dict:
    """Run train-synthetic-test-real: train the classifier on generated data, test on the real fold.

    Generates synthetic signals from the loaded generator, extracts the cached top-20 features,
    trains the stacking classifier on synthetic data, evaluates on the real test fold (and the
    real val split if eval_val), and measures SHAP top-feature overlap with the real model.

    :param cache: trtr cache providing top features, the real test set, and n_train.
    :param model_name: generator model name.
    :param region: 'mouth' or 'nose'.
    :param n_synthetic: number of synthetic signals to generate.
    :param n_jobs: parallel workers.
    :param device: torch device.
    :param init_seed: model/sampling seed.
    :param split_seed: data-split seed.
    :param fold: 1-indexed fold.
    :param run_id: artifact id locating the generator checkpoint.
    :param free_bits, latent_dim, embed_dim, part_embed_dim, lambda_phys, alpha, n_copies:
        hyperparameters recorded in the result.
    :param eval_val: also evaluate on the real validation split as a sanity check.
    :param dataset_dir: dataset root (used only when eval_val).
    :param n_folds: number of cv folds.
    :param cv_mode: 'kfold' or 'loso'.
    :param guidance: classifier-free guidance scale for diffusion (ignored if < 0).
    :param include_subjects: restrict the pool to these subjects.
    :return: result dict with config fields plus 'metrics' (test scores), 'metrics_val'
        (val scores or None), 'feature_overlap', 'top_20_synth_features', and 'trtr_metrics'.
    """
    model, ckpt_stats = load_model(model_name, ckpt_run_id or run_id, device)
    validate_generator(model, cache, model_name, preprocessing)
    if guidance >= 0 and model_name == 'diffusion':
        model.guidance_scale = guidance
    print(f'[TSTR] model={model_name}, n_synthetic={n_synthetic}, preprocessing={preprocessing}')
    # under LOSO the test subject is unseen -> generate from the learned null token
    participant_idx = model.null_part_idx if (cv_mode == 'loso' and getattr(model, '_cond_part', False)) else None
    synth_signals, synth_labels = generate_synthetic_signals(model, model_name, n_synthetic, ckpt_stats, device, init_seed, participant_idx=participant_idx)
    # match the real-data feature space: apply the same per-trial preprocessing to synthetic signals
    synth_signals = preprocess_synth_signals(synth_signals, preprocessing)
    print(f'[TSTR] Generated {n_synthetic} synthetic signals' + (' (null token)' if participant_idx is not None else ''))
    # ensemble: mix in synthetic from a second generator
    if ensemble_model and ensemble_ckpt_run_id:
        m2, stats2 = load_model(ensemble_model, ensemble_ckpt_run_id, device)
        validate_generator(m2, cache, ensemble_model, preprocessing)
        p2 = m2.null_part_idx if (cv_mode == 'loso' and getattr(m2, '_cond_part', False)) else None
        s2_sig, s2_lab = generate_synthetic_signals(m2, ensemble_model, n_synthetic, stats2, device, init_seed, participant_idx=p2)
        s2_sig = preprocess_synth_signals(s2_sig, preprocessing)
        synth_signals = np.concatenate([synth_signals, s2_sig], axis=0)
        synth_labels = np.concatenate([synth_labels, s2_lab], axis=0)
        print(f'[TSTR] ensemble +{len(s2_lab)} synthetic from {ensemble_model} (total {len(synth_labels)})')
    n, _C, T = synth_signals.shape
    ids = np.repeat(np.arange(n), T)
    times = np.tile(np.arange(T), n)
    df_long_synth = pd.DataFrame({'id': ids, 'time': times,
                                  'Humidity': synth_signals[:, 0, :].ravel(),
                                  'Temperature': synth_signals[:, 1, :].ravel()})
    X_synth_raw = extract_fixed_features(df_long_synth, cache['top_20_features_raw'], n_jobs, cache['imputation'])
    X_san = X_synth_raw.copy()
    X_san.columns = [re.sub(r'[^\w]', '_', col) for col in X_san.columns]
    X_synth_top = X_san[cache['top_20_features_sanitized']].values
    stacker_tstr = train_stacking_classifier(X_synth_top, synth_labels, init_seed, n_jobs=n_jobs)
    tstr_metrics = evaluate_classifier(stacker_tstr, cache['X_test_top'], cache['y_test'])
    # validation-split sanity eval
    metrics_val = None
    if eval_val:
        df_val = load_dataset(dataset_dir)
        df_val = df_val[df_val['region'] == region].reset_index(drop=True)
        df_val = preprocess_signals(df_val, preprocessing)
        _, df_val, _ = get_split(df_val, cv_mode=cv_mode, fold=fold - 1, n_folds=n_folds, split_seed=split_seed, include_subjects=include_subjects)
        df_long_val, y_val = df_to_df_long(df_val)
        X_val_raw = extract_fixed_features(df_long_val, cache['top_20_features_raw'], n_jobs, cache['imputation'])
        X_val_san = X_val_raw.copy()
        X_val_san.columns = [re.sub(r'[^\w]', '_', col) for col in X_val_san.columns]
        X_val_top = X_val_san[cache['top_20_features_sanitized']].values
        metrics_val = evaluate_classifier(stacker_tstr, X_val_top, y_val.values)
        print(f"[TSTR] val-split sanity: acc={metrics_val['accuracy']:.3f} f1={metrics_val['f1_weighted']:.3f} (test acc={tstr_metrics['accuracy']:.3f})")
    # fidelity: signal-space MMD of the synthetic set vs the real training trials (classifier input space)
    df_reg = load_dataset(dataset_dir)
    df_reg = preprocess_signals(df_reg[df_reg['region'] == region].reset_index(drop=True), preprocessing)
    df_tr_m, _, df_te_m = get_split(df_reg, cv_mode=cv_mode, fold=fold - 1, n_folds=n_folds, split_seed=split_seed, include_subjects=include_subjects)
    mmd = mmd_report(_signals(df_tr_m), np.array([CLASS_TO_IDX[c] for c in df_tr_m['class']]), synth_signals, synth_labels, _signals(df_te_m))
    xgb_base = stacker_tstr.estimators_[0]
    xgb_clf = xgb_base.named_steps['clf'] if hasattr(xgb_base, 'named_steps') else xgb_base
    explainer = shap.TreeExplainer(xgb_clf)
    shap_values = explainer.shap_values(X_synth_top)
    if shap_values.ndim == 3:
        mean_abs = np.abs(shap_values).mean(axis=0).mean(axis=1)
    else:
        mean_abs = np.abs(shap_values).mean(axis=0)
    k = 10
    top_idx = np.argsort(mean_abs)[::-1][:k]
    top_k_synth = [cache['top_20_features_sanitized'][i] for i in top_idx]
    top_k_real = cache['top_20_features_sanitized'][:k]
    overlap = len(set(top_k_real) & set(top_k_synth)) / k
    return {'training_provenance': model.provenance,
            'model': model_name,
            'region': region,
            'channel': cache.get('channel', 'both'),
            'cv_mode': cv_mode,
            'init_seed': init_seed,
            'split_seed': split_seed,
            'fold': fold,
            'latent_dim': latent_dim,
            'embed_dim': embed_dim,
            'part_embed_dim': part_embed_dim if model_name == 'cvae_part' else None,
            'free_bits': free_bits,
            'lambda_phys': lambda_phys if model_name == 'pinn' else None,
            'alpha': alpha,
            'n_copies': n_copies,
            'n_train_real': cache['n_train'],
            'n_synthetic': n_synthetic,
            'top_20_features': cache['top_20_features_sanitized'],
            'metrics': tstr_metrics,
            'metrics_val': metrics_val,
            'feature_overlap': overlap,
            'mmd': mmd,
            'top_20_synth_features': top_k_synth,
            'trtr_metrics': cache['trtr_metrics']}

def generate_real_aug(df_train: pd.DataFrame, n_aug: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Non-generator augmentation from REAL trials: class-balanced same-class mixup (a convex combo of two
    same-class trials, mixing across subjects within a class to smooth subject boundaries).

    :param df_train: training trials with humidity/temperature/class columns.
    :param n_aug: number of augmented signals to produce.
    :param seed: RNG seed.
    :return: (n_aug, 2, 36) signals and their integer class labels.
    """
    rng = np.random.RandomState(seed)
    # bucket trials by class
    by_class = defaultdict(list)
    for r in df_train.to_dict('records'):
        by_class[CLASS_TO_IDX[r['class']]].append(np.stack([r['humidity'], r['temperature']]).astype(np.float32))
    classes = sorted(by_class)
    # class-balanced mixup
    sigs, labs = [], []
    for i in range(n_aug):
        c = classes[i % len(classes)]
        pool = by_class[c]
        a_sig = pool[rng.randint(len(pool))]
        s = a_sig
        if len(pool) > 1:
            b_sig = pool[rng.randint(len(pool))]
            lam = rng.beta(0.4, 0.4)
            s = lam * a_sig + (1 - lam) * b_sig
        sigs.append(s)
        labs.append(c)
    return np.stack(sigs), np.array(labs)

def generate_jitter_aug(df_train: pd.DataFrame, n_aug: int, alpha: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Non-generator control: Gaussian-jittered copies of REAL trials, replicating the *exact*
    training-time jitter of §4.4.1 (BreathDataset) with NO generative model.

    BreathDataset augments in z-score space: z = (x-mean)/std; z += alpha*N(0,1). Denormalised back
    to physical units this is x + alpha*std_channel*N(0,1), so the perturbation seen by the classifier
    is identical to the one the generator was trained under. Stats are the per-channel std of the
    training fold (matching BreathDataset._compute_stats); sampling is class-balanced like the generator.

    :param df_train: training trials with humidity/temperature/class columns.
    :param n_aug: number of jittered signals to produce.
    :param alpha: Gaussian noise std in z-score space (pass the generator's --alpha, e.g. 0.05).
    :param seed: RNG seed.
    :return: (n_aug, 2, 36) physical-unit signals and their integer class labels.
    """
    rng = np.random.RandomState(seed)
    records = df_train.to_dict('records')
    # per-channel z-score std over the training fold (BreathDataset._compute_stats)
    h_std = np.concatenate([r['humidity'] for r in records]).std()
    t_std = np.concatenate([r['temperature'] for r in records]).std()
    std = np.array([h_std, t_std], dtype=np.float32).reshape(2, 1)
    by_class = defaultdict(list)
    for r in records:
        by_class[CLASS_TO_IDX[r['class']]].append(np.stack([r['humidity'], r['temperature']]).astype(np.float32))
    classes = sorted(by_class)
    sigs, labs = [], []
    for i in range(n_aug):
        c = classes[i % len(classes)]
        pool = by_class[c]
        base = pool[rng.randint(len(pool))]
        s = base + alpha * std * rng.standard_normal(base.shape).astype(np.float32)
        sigs.append(s.astype(np.float32))
        labs.append(c)
    return np.stack(sigs), np.array(labs)

def generate_smote_aug(df_train: pd.DataFrame, n_aug: int, seed: int, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Non-generator control: signal-space SMOTE. Interpolate each new sample toward one of a real
    trial's k nearest same-class neighbours in raw signal space, x_new = x_i + u*(x_nn - x_i), u~U(0,1).

    This is distinct from the feature-space SMOTE already inside the classifier (train_stacking_classifier
    balances the fixed top-20 features): interpolating raw signals and re-extracting features yields points
    off the real-data feature manifold. Class-balanced like the generator; k is clipped to the class size.

    :param df_train: training trials with humidity/temperature/class columns.
    :param n_aug: number of synthetic signals to produce.
    :param seed: RNG seed.
    :param k: neighbourhood size (clipped to per-class count - 1).
    :return: (n_aug, 2, 36) physical-unit signals and their integer class labels.
    """
    rng = np.random.RandomState(seed)
    by_class = defaultdict(list)
    for r in df_train.to_dict('records'):
        by_class[CLASS_TO_IDX[r['class']]].append(np.stack([r['humidity'], r['temperature']]).astype(np.float32))
    classes = sorted(by_class)
    # precompute per-class same-class neighbour indices (self excluded)
    neigh = {}
    for c in classes:
        pool = np.stack(by_class[c])  # (m, 2, 36)
        kc = min(k, len(pool) - 1)
        if kc >= 1:
            nn = NearestNeighbors(n_neighbors=kc + 1).fit(pool.reshape(len(pool), -1))
            neigh[c] = nn.kneighbors(pool.reshape(len(pool), -1), return_distance=False)[:, 1:]
        else:
            neigh[c] = None
    sigs, labs = [], []
    for i in range(n_aug):
        c = classes[i % len(classes)]
        pool = by_class[c]
        j = rng.randint(len(pool))
        base = pool[j]
        if neigh[c] is not None:
            nn_j = neigh[c][j][rng.randint(neigh[c].shape[1])]
            s = base + rng.rand() * (pool[nn_j] - base)
        else:
            s = base
        sigs.append(s.astype(np.float32))
        labs.append(c)
    return np.stack(sigs), np.array(labs)

# train synthetic/real test real
def _signals(df: pd.DataFrame) -> np.ndarray:
    """(n, 2, T) humidity / temperature array of a trial dataframe."""
    return np.stack([np.stack([h, t]) for h, t in zip(df['humidity'], df['temperature'])]).astype(float)

def _stream_alias(model_name: str, ckpt_run_id: str) -> str:
    """Default random-stream name of a physics policy (the generator names of the development runs)."""
    return 'rh15' if model_name == 'rhythm' else ('fp' if '_fp' in ckpt_run_id else 'tp0')

def tstr_plus(cache: dict, model_name: str, region: str, augmentation_ratio: float, n_jobs: int, device, init_seed: int, split_seed: int, fold: int, run_id: str, free_bits: float = 0.0, latent_dim: int = 32, embed_dim: int = 16, part_embed_dim: int = 8, lambda_phys: float = 0.0, alpha: float = 0.0, n_copies: int = 1, cv_mode: str = 'kfold', preprocessing: str = 'raw', ckpt_run_id: str | None = None, real_fraction: float = 1.0, aug_source: str = 'gen', dataset_dir: str = 'dataset', n_folds: int = 5, include_subjects: tuple = (), guidance: float = -1.0, aug_policy: str = 'prior', aug_stream: str = '', eval_val: bool = False) -> dict:
    """Run augmentation TSTR+: train the classifier on real + synthetic data, test on the real fold.

    Generates int(n_train * augmentation_ratio) synthetic signals, concatenates their cached
    top-20 features with the real training features, trains the stacking classifier on the
    combined set, and evaluates on the real test fold.

    :param cache: trtr cache providing real train/test features and n_train.
    :param model_name: generator model name (result model is suffixed with '_plus').
    :param region: 'mouth' or 'nose'.
    :param augmentation_ratio: synthetic-to-real ratio for the augmentation set.
    :param n_jobs: parallel workers.
    :param device: torch device.
    :param init_seed: model/sampling seed.
    :param split_seed: data-split seed.
    :param fold: 1-indexed fold.
    :param run_id: artifact id locating the generator checkpoint.
    :param free_bits, latent_dim, embed_dim, part_embed_dim, lambda_phys, alpha, n_copies:
        hyperparameters recorded in the result.
    :param cv_mode: 'kfold' or 'loso'.
    :param aug_policy: 'prior' (generator prior samples) or a physics policy of core/augment.py (steer_margin,
        steer_knn, obsdr) built from the generator and the real training trials.
    :param aug_stream: random-stream name of the policy ('' = default per generator).
    :param eval_val: also score the augmented and the real-only stacker on the real validation split.
    :return: result dict with config fields plus 'augmentation_ratio', 'metrics' (test scores),
        'metrics_realonly' (matched real-only stacker), 'metrics_val' / 'metrics_realonly_val' (or None),
        'mmd' (signal-space MMD of the synthetic set, see core.augment.mmd_report), 'feature_overlap' (None),
        'top_20_synth_features' (None), and 'trtr_metrics'.
    """
    if real_fraction != 1.0:
        raise ValueError('Scarcity evaluation requires retraining the generator and feature selector on the same subset; real_fraction must be 1')
    X_real, y_real = cache['X_train_top'], cache['y_train']
    if real_fraction < 1.0:
        idx = np.arange(len(y_real))
        keep, _ = train_test_split(idx, train_size=real_fraction, stratify=y_real, random_state=init_seed)
        X_real, y_real = X_real[keep], y_real[keep]
    n_real_used = len(y_real)
    n_synthetic = int(n_real_used * augmentation_ratio)
    df = load_dataset(dataset_dir)
    df = df[df['region'] == region].reset_index(drop=True)
    df_tr, df_va, df_te = get_split(df, cv_mode=cv_mode, fold=fold - 1, n_folds=n_folds, split_seed=split_seed, include_subjects=include_subjects)
    stream = None
    if aug_source == 'gen':
        model, ckpt_stats = load_model(model_name, ckpt_run_id or run_id, device)
        validate_generator(model, cache, model_name, preprocessing)
        if model_name == 'diffusion' and guidance >= 0:
            model.guidance_scale = guidance
        print(f'[TSTR+] model={model_name}, n_synthetic={n_synthetic}, n_real_used={n_real_used} (frac={real_fraction}), preprocessing={preprocessing}')
        participant_idx = model.null_part_idx if (cv_mode == 'loso' and getattr(model, '_cond_part', False)) else None
        if aug_policy == 'prior':
            synth_signals, synth_labels = generate_synthetic_signals(model, model_name, n_synthetic, ckpt_stats, device, init_seed, participant_idx=participant_idx)
        else:
            if model_name not in ('tpinn', 'rhythm'):
                raise ValueError('physics augmentation policies need a tpinn or rhythm generator')
            stream = f'{aug_stream or _stream_alias(model_name, ckpt_run_id or run_id)}/{POLICY_KINDS[aug_policy]}:{augmentation_ratio}'
            synth_signals, synth_labels = augment(aug_policy, model, model_name, ckpt_stats, df_tr, cache, n_synthetic, init_seed, fold, stream,
                                                  lambda signals: top_features(signals, cache, preprocessing), cv_mode)
            print(f'[TSTR+] policy={aug_policy} stream={stream}')
    else:
        # non-generator REAL-signal augmentation controls (no generative model, no checkpoint loaded)
        if aug_source == 'mixup':
            synth_signals, synth_labels = generate_real_aug(df_tr, n_synthetic, init_seed)
        elif aug_source == 'jitter':
            if alpha <= 0:
                raise ValueError('aug_source=\'jitter\' needs --alpha > 0 (the §4.4.1 jitter magnitude, e.g. 0.05)')
            synth_signals, synth_labels = generate_jitter_aug(df_tr, n_synthetic, alpha, init_seed)
        elif aug_source == 'smote':
            synth_signals, synth_labels = generate_smote_aug(df_tr, n_synthetic, init_seed)
        else:
            raise ValueError(f'unknown aug_source: {aug_source!r}')
        print(f'[TSTR+] real-aug {aug_source} (alpha={alpha}), n={n_synthetic}, n_real_used={n_real_used}, preprocessing={preprocessing}')
    # match the real-data feature space (cache X_train/X_test are already preprocessed in trtr)
    synth_signals = preprocess_synth_signals(synth_signals, preprocessing)
    n, _C, T = synth_signals.shape
    ids = np.repeat(np.arange(n), T)
    times = np.tile(np.arange(T), n)
    df_long_synth = pd.DataFrame({'id': ids, 'time': times,
                                  'Humidity': synth_signals[:, 0, :].ravel(),
                                  'Temperature': synth_signals[:, 1, :].ravel()})
    X_synth_raw = extract_fixed_features(df_long_synth, cache['top_20_features_raw'], n_jobs, cache['imputation'])
    X_san = X_synth_raw.copy()
    X_san.columns = [re.sub(r'[^\w]', '_', col) for col in X_san.columns]
    X_synth_top = X_san[cache['top_20_features_sanitized']].values
    n_synth_kept = len(synth_labels)
    X_combined = np.concatenate([X_real, X_synth_top])
    y_combined = np.concatenate([y_real, synth_labels])
    stacker = train_stacking_classifier(X_combined, y_combined, init_seed, n_jobs=n_jobs)
    metrics = evaluate_classifier(stacker, cache['X_test_top'], cache['y_test'])
    # matched real-only-at-fraction baseline (same subsample) so the augmentation LIFT is paired per fold
    stacker_ro = train_stacking_classifier(X_real, y_real, init_seed, n_jobs=n_jobs)
    metrics_realonly = evaluate_classifier(stacker_ro, cache['X_test_top'], cache['y_test'])
    metrics_val = metrics_realonly_val = None
    if eval_val:
        dl_val, y_val = df_to_df_long(preprocess_signals(df_va, preprocessing))
        X_val_top = _feat_top(dl_val, cache, n_jobs)
        metrics_val = evaluate_classifier(stacker, X_val_top, y_val.values)
        metrics_realonly_val = evaluate_classifier(stacker_ro, X_val_top, y_val.values)
    # fidelity: signal-space MMD of the synthetic set vs the real training trials (classifier input space)
    y_tr = np.array([CLASS_TO_IDX[c] for c in df_tr['class']])
    mmd = mmd_report(_signals(preprocess_signals(df_tr, preprocessing)), y_tr, synth_signals, synth_labels, _signals(preprocess_signals(df_te, preprocessing)), with_test=(cv_mode == 'kfold'))
    print(f"[TSTR+] aug acc={metrics['accuracy']:.4f} | real-only@frac acc={metrics_realonly['accuracy']:.4f} | lift={metrics['accuracy']-metrics_realonly['accuracy']:+.4f} | MMD {mmd['mmd']:.4f} (real test vs train {mmd['mmd_real_ref']:.4f})")
    return {'training_provenance': model.provenance if aug_source == 'gen' else None,
            'model': f'{model_name}_plus',
            'region': region,
            'channel': cache.get('channel', 'both'),
            'cv_mode': cv_mode,
            'init_seed': init_seed,
            'split_seed': split_seed,
            'fold': fold,
            'latent_dim': latent_dim,
            'embed_dim': embed_dim,
            'part_embed_dim': part_embed_dim if model_name == 'cvae_part' else None,
            'free_bits': free_bits,
            'lambda_phys': lambda_phys if model_name == 'pinn' else None,
            'alpha': alpha,
            'n_copies': n_copies,
            'n_train_real': cache['n_train'],
            'n_real_used': n_real_used,
            'real_fraction': real_fraction,
            'n_synthetic': n_synthetic,
            'n_synth_kept': n_synth_kept,
            'augmentation_ratio': augmentation_ratio,
            'aug_source': aug_source,
            'aug_policy': aug_policy,
            'aug_stream': stream,
            'metrics_realonly': metrics_realonly,
            'metrics_val': metrics_val,
            'metrics_realonly_val': metrics_realonly_val,
            'mmd': mmd,
            'top_20_features': cache['top_20_features_sanitized'],
            'metrics': metrics,
            'feature_overlap': None,
            'top_20_synth_features': None,
            'trtr_metrics': cache['trtr_metrics']}

def _feat_top(df_long: pd.DataFrame, cache: dict, n_jobs: int) -> np.ndarray:
    """Extract the cached top-20 features from long-format data and return the sanitized-ordered matrix."""
    X = extract_fixed_features(df_long, cache['top_20_features_raw'], n_jobs, cache['imputation'])
    X.columns = [re.sub(r'[^\w]', '_', c) for c in X.columns]
    return X[cache['top_20_features_sanitized']].values

def build_run_id(model: str, region: str, init_seed: int, fold: int, latent_dim: int, embed_dim: int, part_embed_dim: int, free_bits: float, alpha: float, n_copies: int, channel: str = 'both', lambda_phys: float = 1.0, cv_mode: str = 'kfold', part_dropout: float = 0.0, loso_tag: str = '', phys_residual: bool = False, class_transport: bool = False, parametric_source: bool = False, learn_cir_params: bool = False, ode: bool = False, subj_adv_lambda: float = 0.0, diff_hidden: int = 64, n_steps: int = 200, gan_hidden: int = 64, gan_loss: str = 'bce', gan_lr_d: float | None = None, lr: float = 1e-3, include_subjects: tuple = (), cir_tag: str = '', preprocessing: str = 'raw', phys_prep: str = 'peakscale', hp_tag: str = '', aug_ratio: float = 1.0, real_fraction: float = 1.0, ensemble_model: str = '', aug_source: str = 'gen', guidance: float = -1.0, n_synthetic: int | None = None, classwise_vae: bool = False, temperature_tau: float | None = None, aug_policy: str = 'prior') -> str:
    """Build the generator artifact id encoding the model and its config flags.

    Assembles a per-model-family base id (region, seed, dims, model-specific markers) and appends
    fold, cv-mode, dropout, subset, cir, preprocessing, jitter, channel, and hp tags.

    :return: the run-id string used to locate checkpoints and result files.
    """
    base_model = model.removesuffix('_plus')
    if temperature_tau is not None and (base_model != 'tpinn' or ode or class_transport or parametric_source):
        raise ValueError('temperature_tau requires the fixed-sensor tpinn transport branch')
    if base_model == 'vae':
        run_id = f'{region}_s{init_seed}_ld{latent_dim}_fb{free_bits}'
    elif base_model == 'cvae':
        run_id = f'{region}_s{init_seed}_ld{latent_dim}_ed{embed_dim}_fb{free_bits}'
    elif base_model == 'cvae_part':
        run_id = f'{region}_s{init_seed}_ld{latent_dim}_ed{embed_dim}_pd{part_embed_dim}_fb{free_bits}' + (f'_adv{subj_adv_lambda}' if subj_adv_lambda > 0 else '')
    elif base_model == 'pinn':
        run_id = f'{region}_s{init_seed}_ld{latent_dim}_ed{embed_dim}_phys{lambda_phys}'
    elif base_model == 'tpinn':
        run_id = f'{region}_s{init_seed}_ld{latent_dim}_ed{embed_dim}_tphys' + ('_ct' if class_transport else '') + ('_ps' if parametric_source else '') + ('_res' if phys_residual else '') + ('_learn' if learn_cir_params else '') + ('_ode' if ode else '')
    elif base_model == 'rhythm':
        run_id = f'{region}_s{init_seed}_rhythm'
    elif base_model == 'diffusion':
        run_id = f'{region}_s{init_seed}_ed{embed_dim}_diff_h{diff_hidden}_st{n_steps}'
    elif base_model == 'gan':
        run_id = f'{region}_s{init_seed}_ld{latent_dim}_ed{embed_dim}_gan_h{gan_hidden}_{gan_loss}_lrd{gan_lr_d if gan_lr_d is not None else lr}'
    else:
        run_id = f'{region}_s{init_seed}'
    run_id += f'_f{fold}{_cv_marker(cv_mode)}{loso_tag}{_drop_marker(part_dropout)}{subset_tag(include_subjects)}{cir_marker(cir_tag)}{prep_marker(preprocessing)}{phys_prep_marker(phys_prep)}'
    if alpha > 0:
        run_id += f'_a{alpha}_n{n_copies}'
    # eval-only marker: augmentation ratio (tstr_plus). Default 1.0 -> no marker (checkpoint id keeps 1.0,
    # so committed generators are reused; only the result run_id / json filename is namespaced by ratio).
    if aug_ratio != 1.0:
        run_id += f'_augr{aug_ratio}'
    if real_fraction != 1.0:
        run_id += f'_rf{real_fraction}'
    if ensemble_model:
        run_id += f'_ens{ensemble_model}'
    if aug_source != 'gen':
        run_id += f'_aug{aug_source}'
    if aug_policy != 'prior':
        run_id += f'_{aug_policy}'
    if channel != 'both':
        run_id += f'_ch{channel}'
    if guidance >= 0:
        run_id += f'_g{guidance}'
    if n_synthetic is not None:
        run_id += f'_ns{n_synthetic}'
    if hp_tag:
        run_id += f'_{hp_tag}'
    run_id += sensor_marker(temperature_tau)
    if classwise_vae:
        run_id += '_classwise'
    return run_id

# main
def main():
    """Run the TRTR/TSTR/TSTR+ pipeline (or aggregate-only summary merge) from CLI args."""
    args = parse_args()
    device = torch.device('cpu')
    # nested-LOSO: 0-indexed excludes from the 1-indexed CLI
    exclude_subjects = tuple(int(x) - 1 for x in args.loso_exclude.split(',') if x.strip()) if args.loso_exclude else ()
    loso_tag = loso_path_tag(args.loso_trial_val, exclude_subjects)
    include_subjects = tuple(x.strip() for x in args.include_subjects.split(',') if x.strip())
    sub_tag = subset_tag(include_subjects)
    sub_value = '+'.join(sorted(include_subjects))
    phys_variant = '_'.join(k for k, v in (('res', args.phys_residual), ('ct', args.class_transport), ('ps', args.parametric_source), ('learn', args.learn_cir_params)) if v)
    prep_tag = prep_marker(args.preprocessing)
    # aggregate mode
    if args.aggregate:
        regions = args.regions.split(',') if args.regions else [args.region]
        init_seeds = [int(s) for s in args.init_seeds.split(',')] if args.init_seeds else [args.init_seed]
        folds = [int(s) for s in args.folds.split(',')] if args.folds else [args.fold]
        n_rows = 0
        for region in regions:
            for init_seed in init_seeds:
                for fold in folds:
                    rid = build_run_id(args.model, region, init_seed, fold, args.latent_dim, args.embed_dim, args.part_embed_dim, args.free_bits, args.alpha, args.n_copies, args.channel, args.lambda_phys, args.cv_mode, args.part_dropout, loso_tag, phys_residual=args.phys_residual, class_transport=args.class_transport, parametric_source=args.parametric_source, learn_cir_params=args.learn_cir_params, ode=args.ode, subj_adv_lambda=args.subj_adv_lambda, diff_hidden=args.diff_hidden, n_steps=args.n_steps, gan_hidden=args.gan_hidden, gan_loss=args.gan_loss, gan_lr_d=args.gan_lr_d, lr=args.lr, include_subjects=include_subjects, cir_tag=args.cir_tag, preprocessing=args.preprocessing, phys_prep=args.phys_prep, hp_tag=args.hp_tag, aug_ratio=(args.augmentation_ratio if args.mode == 'tstr_plus' else 1.0), real_fraction=args.real_fraction, ensemble_model=args.ensemble_model, aug_source=args.aug_source, guidance=args.guidance, n_synthetic=args.n_synthetic, classwise_vae=args.classwise_vae, temperature_tau=args.temperature_tau, aug_policy=(args.aug_policy if args.mode == 'tstr_plus' else 'prior'))
                    model_dir = args.model + ('_plus' if args.mode == 'tstr_plus' else '')
                    path = f'results/{model_dir}/{rid}_tstr.json'
                    if not os.path.exists(path):
                        print(f'  [AGGREGATE] missing {path}, skipping')
                        continue
                    with open(path) as f:
                        r = json.load(f)
                    r['phys_variant'] = phys_variant
                    r['cir'] = args.cir_tag
                    r['prep'] = args.preprocessing
                    r['phys_prep'] = args.phys_prep
                    save_summary(r)
                    n_rows += 1
        print(f'[AGGREGATE] merged {n_rows} rows into results/summary.csv')
        return
    if args.region is None:
        raise SystemExit('[TSTR] --region is required (except with --aggregate)')
    run_id = build_run_id(args.model, args.region, args.init_seed, args.fold, args.latent_dim, args.embed_dim, args.part_embed_dim, args.free_bits, args.alpha, args.n_copies, args.channel, args.lambda_phys, args.cv_mode, args.part_dropout, loso_tag, phys_residual=args.phys_residual, class_transport=args.class_transport, parametric_source=args.parametric_source, learn_cir_params=args.learn_cir_params, ode=args.ode, subj_adv_lambda=args.subj_adv_lambda, diff_hidden=args.diff_hidden, n_steps=args.n_steps, gan_hidden=args.gan_hidden, gan_loss=args.gan_loss, gan_lr_d=args.gan_lr_d, lr=args.lr, include_subjects=include_subjects, cir_tag=args.cir_tag, preprocessing=args.preprocessing, phys_prep=args.phys_prep, hp_tag=args.hp_tag, aug_ratio=(args.augmentation_ratio if args.mode == 'tstr_plus' else 1.0), real_fraction=(args.real_fraction if args.mode == 'tstr_plus' else 1.0), ensemble_model=args.ensemble_model, aug_source=(args.aug_source if args.mode == 'tstr_plus' else 'gen'), guidance=args.guidance, n_synthetic=args.n_synthetic, classwise_vae=args.classwise_vae, temperature_tau=args.temperature_tau, aug_policy=(args.aug_policy if args.mode == 'tstr_plus' else 'prior'))
    ckpt_run_id = build_run_id(args.model, args.region, args.init_seed, args.fold, args.latent_dim, args.embed_dim, args.part_embed_dim, args.free_bits, args.alpha, args.n_copies, 'both', args.lambda_phys, args.cv_mode, args.part_dropout, loso_tag, phys_residual=args.phys_residual, class_transport=args.class_transport, parametric_source=args.parametric_source, learn_cir_params=args.learn_cir_params, ode=args.ode, subj_adv_lambda=args.subj_adv_lambda, diff_hidden=args.diff_hidden, n_steps=args.n_steps, gan_hidden=args.gan_hidden, gan_loss=args.gan_loss, gan_lr_d=args.gan_lr_d, lr=args.lr, include_subjects=include_subjects, cir_tag=args.cir_tag, preprocessing='raw', phys_prep=args.phys_prep, hp_tag=args.hp_tag, classwise_vae=args.classwise_vae, temperature_tau=args.temperature_tau)
    # ensemble second-generator checkpoint id (tpinn = tpinn-res @ stdscale, the champion physics model)
    ensemble_ckpt_run_id = None
    if args.ensemble_model == 'tpinn':
        ensemble_ckpt_run_id = build_run_id('tpinn', args.region, args.init_seed, args.fold, args.latent_dim, args.embed_dim, args.part_embed_dim, args.free_bits, args.alpha, args.n_copies, 'both', cv_mode=args.cv_mode, part_dropout=args.part_dropout, loso_tag=loso_tag, phys_residual=True, phys_prep='stdscale', preprocessing='raw', include_subjects=include_subjects, hp_tag=args.hp_tag)
    # reproducibility
    seed_everything(args.init_seed)
    tag = '[TRTR]' if args.model == 'trtr' else '[TSTR]'
    print(f'{tag} init_seed={args.init_seed} split_seed={args.split_seed} fold={args.fold} cv={args.cv_mode}{loso_tag} | Model: {args.model} | Region: {args.region} | Device: {device}')
    # train-real-test-real
    # build cache
    if args.force_rebuild:
        path = _cache_path(args.region, args.init_seed, args.split_seed, args.fold, args.n_folds, args.channel, args.cv_mode, loso_tag, sub_tag, prep_tag)
        if os.path.exists(path):
            os.remove(path)
            print('[TRTR] Removed cache.')
    # load cache
    cache = load_cache(args.region, args.init_seed, args.split_seed, args.fold, args.n_folds, args.channel, args.cv_mode, loso_tag, sub_tag, prep_tag, args.dataset_dir)
    if cache is None:
        cache = trtr(args.dataset_dir, args.region, args.n_jobs, args.init_seed, args.split_seed, args.fold, args.n_folds, args.channel, args.cv_mode, exclude_subjects, args.loso_trial_val, include_subjects, args.preprocessing)
        print('[TRTR] Built cache.')
    else:
        print('[TRTR] Loaded cache.')
    # save trtr results
    if args.model == 'trtr':
        result = {'model': 'trtr',
                  'region': args.region,
                  'channel': args.channel,
                  'cv_mode': args.cv_mode,
                  'init_seed': args.init_seed,
                  'split_seed': args.split_seed,
                  'fold': args.fold,
                  'n_train_real': cache['n_train'],
                  'n_synthetic': cache['n_train'],
                  'top_20_features': cache['top_20_features_sanitized'],
                  'metrics': cache['trtr_metrics'],
                  'feature_overlap': None,
                  'top_20_synth_features': None,
                  'trtr_metrics': cache['trtr_metrics']}
        result['subset'] = sub_value
        result['phys_variant'] = phys_variant
        result['cir'] = args.cir_tag
        result['prep'] = args.preprocessing
        result['phys_prep'] = args.phys_prep
        result['provenance'] = cache['provenance']
        result['evaluation_config'] = {k: v for k, v in vars(args).items() if k not in ('dataset_dir', 'force_rebuild', 'no_summary', 'aggregate')}
        save_result(result, args.model, run_id)
        if not args.no_summary:
            save_summary(result)
        return
    # train-synthetic-test-real
    if args.mode == 'tstr_plus':
        result = tstr_plus(cache, args.model, args.region, args.augmentation_ratio, args.n_jobs, device, args.init_seed, args.split_seed, args.fold, run_id, args.free_bits, args.latent_dim, args.embed_dim, args.part_embed_dim, args.lambda_phys, args.alpha, args.n_copies, cv_mode=args.cv_mode, preprocessing=args.preprocessing, ckpt_run_id=ckpt_run_id, real_fraction=args.real_fraction, aug_source=args.aug_source, dataset_dir=args.dataset_dir, n_folds=args.n_folds, include_subjects=include_subjects, guidance=args.guidance, aug_policy=args.aug_policy, aug_stream=args.aug_stream, eval_val=args.eval_val)
        result['subset'] = sub_value
        result['phys_variant'] = phys_variant
        result['cir'] = args.cir_tag
        result['prep'] = args.preprocessing
        result['phys_prep'] = args.phys_prep
        result['provenance'] = cache['provenance']
        result['evaluation_config'] = {k: v for k, v in vars(args).items() if k not in ('dataset_dir', 'force_rebuild', 'no_summary', 'aggregate')}
        save_result(result, result['model'], run_id)
        if not args.no_summary:
            save_summary(result)
    else:
        n_synthetic = args.n_synthetic if args.n_synthetic is not None else cache['n_train']
        result = tstr(cache, args.model, args.region, n_synthetic, args.n_jobs, device, args.init_seed, args.split_seed, args.fold, run_id, args.free_bits, args.latent_dim, args.embed_dim, args.part_embed_dim, args.lambda_phys, args.alpha, args.n_copies, eval_val=args.eval_val, dataset_dir=args.dataset_dir, n_folds=args.n_folds, cv_mode=args.cv_mode, guidance=args.guidance, include_subjects=include_subjects, preprocessing=args.preprocessing, ckpt_run_id=ckpt_run_id, ensemble_model=args.ensemble_model, ensemble_ckpt_run_id=ensemble_ckpt_run_id)
        print(f"[TSTR] test accuracy={result['metrics']['accuracy']:.4f}")
        result['subset'] = sub_value
        result['phys_variant'] = phys_variant
        result['cir'] = args.cir_tag
        result['prep'] = args.preprocessing
        result['phys_prep'] = args.phys_prep
        result['provenance'] = cache['provenance']
        result['evaluation_config'] = {k: v for k, v in vars(args).items() if k not in ('dataset_dir', 'force_rebuild', 'no_summary', 'aggregate')}
        save_result(result, args.model, run_id)
        if not args.no_summary:
            save_summary(result)

if __name__ == '__main__':
    main()
