import argparse
import json
import os
import pickle
import re

import numpy as np
import pandas as pd
import torch
from tsfresh import extract_features, extract_relevant_features
from tsfresh.feature_extraction.settings import from_columns
from tsfresh.utilities.dataframe_functions import impute
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
import shap
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from data import CLASSES, CLASS_TO_IDX, BreathDataset, load_dataset, split_dataset
from models.vae import CVAE, VAE

CACHE_DIR = 'results/tstr'

# cli
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='TSTR evaluation for breath signal generative models')
    p.add_argument('--model', required=True, choices=['trtr', 'vae', 'cvae'])
    p.add_argument('--region', required=True, choices=['mouth', 'nose'])
    p.add_argument('--n_synthetic', type=int, default=None, help='Number of synthetic samples (default: match real train size)')
    p.add_argument('--dataset_dir', default='dataset')
    p.add_argument('--n_jobs', type=int, default=4)
    p.add_argument('--force_rebuild', action='store_true', help='Delete existing Phase 1 cache and rebuild from scratch')
    return p.parse_args()

# utils
def df_to_df_long(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    hum_all  = np.stack(df['humidity'].values)
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

def extract_fixed_features(df_long: pd.DataFrame, top_features_raw: list[str], n_jobs: int = 4) -> pd.DataFrame:
    kind_to_fc = from_columns(top_features_raw)
    X = extract_features(df_long, column_id='id', column_sort='time', kind_to_fc_parameters=kind_to_fc, n_jobs=n_jobs)
    impute(X)
    missing = [c for c in top_features_raw if c not in X.columns]
    if missing:
        for c in missing:
            X[c] = 0.0
    return X[top_features_raw]

def train_stacking_classifier(X_train: np.ndarray, y_train: np.ndarray) -> StackingClassifier:
    min_class = int(np.bincount(y_train).min())
    k_neighbors = min(5, min_class - 1)
    if k_neighbors < 1:
        X_res, y_res = X_train, y_train
    else:
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_res, y_res = smote.fit_resample(X_train, y_train)
    
    xgb_clf = XGBClassifier(eval_metric='mlogloss', random_state=42, max_depth=4, reg_alpha=0.5, reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8, n_estimators=300,)
    cat_clf = CatBoostClassifier(logging_level='Silent', random_state=42, iterations=300, depth=4, l2_leaf_reg=5.0, random_strength=2.0, bagging_temperature=2.0, od_type='Iter', od_wait=20,)
    meta_clf = RandomForestClassifier(n_estimators=150, max_depth=3, min_samples_leaf=5, min_samples_split=10, random_state=42,)
    stacker = StackingClassifier(estimators=[('xgb', xgb_clf), ('cat', cat_clf)], final_estimator=meta_clf, passthrough=True, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), n_jobs=-1,)
    stacker.fit(X_res, y_res)
    
    return stacker

def evaluate_classifier(clf: StackingClassifier, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    try:
        roc_auc = float(roc_auc_score(y_test, y_prob, multi_class='ovr'))
    except Exception as e:
        print(f'ROC-AUC failed: {e}')
        roc_auc = float('nan')

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    return {'accuracy':    float(accuracy_score(y_test, y_pred)),
            'f1_weighted': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
            'roc_auc_ovr': roc_auc,
            'log_loss':    float(log_loss(y_test, y_prob)),
            'per_class_f1': {cls: float(report.get(str(i), {}).get('f1-score', float('nan'))) for i, cls in enumerate(CLASSES)}}

def save_result(result: dict, model: str, region: str) -> None:
    path = os.path.join(CACHE_DIR, f'{model}_{region}.json')

    def _json_safe(obj):
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
    with open(path, 'w') as f:
        json.dump(_json_safe(result), f, indent=2)

def save_summary(result: dict) -> None:
    csv_path = os.path.join(CACHE_DIR, 'summary.csv')
    m = result['metrics']
    new_row = {'model': result['model'],
               'region': result['region'],
               'accuracy': m['accuracy'],
               'f1_weighted': m['f1_weighted'],
               'roc_auc_ovr': m['roc_auc_ovr'],
               'log_loss': m['log_loss'],
               'f1_bradypnea': m['per_class_f1']['bradypnea'],
               'f1_eupnea': m['per_class_f1']['eupnea'],
               'f1_tachypnea': m['per_class_f1']['tachypnea'],
               'feature_overlap': result.get('feature_overlap'),
               'n_synthetic': result.get('n_synthetic', result['n_train_real'])}
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        df_new = pd.concat([df_old, pd.DataFrame([new_row])], ignore_index=True)
        df_new = df_new.drop_duplicates(subset=['model', 'region'], keep='last')
    else:
        df_new = pd.DataFrame([new_row])
    df_new.to_csv(csv_path, index=False)

# train real test real
def _cache_path(region: str) -> str:
    return os.path.join(CACHE_DIR, f'{region}_cache.pkl')

def load_cache(region: str) -> dict | None:
    path = _cache_path(region)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

def trtr(dataset_dir: str, region: str, n_jobs: int) -> dict:
    # load dataset
    df = load_dataset(dataset_dir)
    df = df[df['region'] == region].reset_index(drop=True)
    df_train, df_val, df_test = split_dataset(df, random_state=42)
    stats = BreathDataset(df_train).stats

    ## train real
    df_long_train, y_train = df_to_df_long(df_train)

    # feature extraction
    X_train_raw = extract_relevant_features(df_long_train, y_train, column_id='id', column_sort='time', n_jobs=n_jobs)
    impute(X_train_raw)

    # build name mapping for sanitization
    raw_to_san = {col: re.sub(r'[^\w]', '_', col) for col in X_train_raw.columns}
    san_to_raw = {v: k for k, v in raw_to_san.items()}
    X_full_san = X_train_raw.copy()
    X_full_san.columns = [raw_to_san[c] for c in X_full_san.columns]

    # lgbm and shap feature importance
    X_tr, X_val, y_tr, y_val = train_test_split(X_full_san, y_train, test_size=0.2, stratify=y_train, random_state=42)
    param_grid = {'max_depth': [4, 6],
                  'reg_alpha': [0.1, 1.0],
                  'reg_lambda': [0.5, 1.0],
                  'colsample_bytree': [0.8, 1.0]}
    base_lgbm = LGBMClassifier(n_estimators=1000, learning_rate=0.05, random_state=42, verbose=-1)
    gs = GridSearchCV(base_lgbm, param_grid, cv=StratifiedKFold(3), scoring='accuracy', n_jobs=-1, verbose=0)
    gs.fit(X_tr, y_tr)

    best_lgbm = LGBMClassifier(**gs.best_params_, n_estimators=1000, learning_rate=0.05, random_state=42, verbose=-1)
    best_lgbm.fit(X_tr.values, y_tr.values, eval_set=[(X_val.values, y_val.values)], eval_metric='multi_logloss', callbacks=[early_stopping(50, verbose=False), log_evaluation(0)])

    explainer = shap.TreeExplainer(best_lgbm)
    shap_values = explainer.shap_values(X_full_san.values)

    mean_abs = np.abs(shap_values).mean(axis=0).mean(axis=1)
    top_idx = np.argsort(mean_abs)[::-1][:20]
    top_20_san = X_full_san.columns.to_numpy()[top_idx].tolist()
    top_20_raw = [san_to_raw[s] for s in top_20_san]
    X_train_top = X_full_san[top_20_san].values 

    ## test real
    df_long_test, y_test = df_to_df_long(df_test)
    X_test_raw_top = extract_fixed_features(df_long_test, top_20_raw, n_jobs)
    X_test_san = X_test_raw_top.copy()
    X_test_san.columns = [re.sub(r'[^\w]', '_', col) for col in X_test_san.columns]
    X_test_top = X_test_san[top_20_san].values

    # train stack classifier
    clf = train_stacking_classifier(X_train_top, y_train.values)

    # evaluate classifier
    trtr_metrics = evaluate_classifier(clf, X_test_top, y_test.values)

    # build cache
    cache = {'top_20_features_raw': top_20_raw,
             'top_20_features_sanitized': top_20_san,
             'X_train_top': X_train_top,
             'X_test_top': X_test_top,
             'y_train': y_train.values,
             'y_test': y_test.values,
             'stats': stats,
             'n_train': len(df_train),
             'trtr_metrics': trtr_metrics}
    with open(_cache_path(region), 'wb') as f:
        pickle.dump(cache, f)

    return cache

# train synthetic test real
def load_model(model_name: str, region: str, device: 'torch.device'):
    ckpt_path = f'checkpoints/{model_name}_{region}.pt'
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}\n'
                                f'Run: python train_{model_name}.py --region {region}')

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if model_name == 'vae':
        model = VAE(latent_dim=ckpt['latent_dim'])
    elif model_name == 'cvae':
        model = CVAE(latent_dim=ckpt['latent_dim'], embed_dim=ckpt['embed_dim'])
    else:
        raise ValueError(f'Unknown model: {model_name}')

    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()
    return model, ckpt['stats']

def generate_synthetic_signals(model, model_name: str, n_synthetic: int, stats: dict, device: 'torch.device') -> tuple[np.ndarray, np.ndarray]:
    n_classes = len(CLASSES)
    n_per_class = n_synthetic // n_classes
    remainder = n_synthetic % n_classes
    counts = [n_per_class + (1 if i < remainder else 0) for i in range(n_classes)]

    mean_t = torch.tensor(stats['mean'], dtype=torch.float32).view(1, 2, 1).to(device)
    std_t  = torch.tensor(stats['std'], dtype=torch.float32).view(1, 2, 1).to(device)

    if model_name == 'vae':
        with torch.no_grad():
            z = model.sample(n_synthetic, device)  # (n, 2, 36) z-scored
        signals_phys = (z * std_t + mean_t).cpu().numpy()
        labels = np.concatenate([np.full(c, i) for i, c in enumerate(counts)])

    elif model_name == 'cvae':
        all_signals, all_labels = [], []
        for cls_idx, count in enumerate(counts):
            y_cls = torch.tensor(cls_idx, dtype=torch.long)
            with torch.no_grad():
                z_cls = model.sample(count, y_cls, device)   # (count, 2, 36) z-scored
            all_signals.append((z_cls * std_t + mean_t).cpu().numpy())
            all_labels.append(np.full(count, cls_idx))
        signals_phys = np.concatenate(all_signals, axis=0)
        labels = np.concatenate(all_labels,  axis=0)

    return signals_phys, labels

def tstr(cache: dict, model_name: str, region: str, n_synthetic: int, n_jobs: int, device) -> dict:
    model, ckpt_stats = load_model(model_name, region, device)
    print(f'[TSTR] model={model_name}, n_synthetic={n_synthetic}')

    synth_signals, synth_labels = generate_synthetic_signals(model, model_name, n_synthetic, ckpt_stats, device)
    print(f'[TSTR] Generated {n_synthetic} synthetic signals')

    n, _C, T = synth_signals.shape
    ids = np.repeat(np.arange(n), T)
    times = np.tile(np.arange(T), n)
    df_long_synth = pd.DataFrame({'id': ids, 'time': times,
                                  'Humidity': synth_signals[:, 0, :].ravel(),
                                  'Temperature': synth_signals[:, 1, :].ravel()})

    X_synth_raw = extract_fixed_features(df_long_synth, cache['top_20_features_raw'], n_jobs)
    X_san = X_synth_raw.copy()
    X_san.columns = [re.sub(r'[^\w]', '_', col) for col in X_san.columns]
    X_synth_top = X_san[cache['top_20_features_sanitized']].values

    stacker_tstr = train_stacking_classifier(X_synth_top, synth_labels)
    tstr_metrics = evaluate_classifier(stacker_tstr, cache['X_test_top'], cache['y_test'])

    # feature overlap: top-10 of synth SHAP vs real top-10
    # cache['top_20_features_sanitized'] is already ranked by real SHAP importance
    xgb_clf = stacker_tstr.estimators_[0]
    explainer = shap.TreeExplainer(xgb_clf)
    shap_values = explainer.shap_values(X_synth_top)
    if shap_values.ndim == 3:
        mean_abs = np.abs(shap_values).mean(axis=0).mean(axis=1)
    else:
        mean_abs = np.abs(shap_values).mean(axis=0)

    k = 10
    top_idx = np.argsort(mean_abs)[::-1][:k]
    top_k_synth = [cache['top_20_features_sanitized'][i] for i in top_idx]
    top_k_real  = cache['top_20_features_sanitized'][:k]
    overlap = len(set(top_k_real) & set(top_k_synth)) / k

    return {'model':                 model_name,
            'region':                region,
            'n_train_real':          cache['n_train'],
            'n_synthetic':           n_synthetic,
            'top_20_features':       cache['top_20_features_sanitized'],
            'metrics':               tstr_metrics,
            'feature_overlap':       overlap,
            'top_20_synth_features': top_k_synth,
            'trtr_metrics':          cache['trtr_metrics']}

# main
def main():
    args = parse_args()
    os.makedirs(CACHE_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps'  if torch.backends.mps.is_available() else 'cpu')
    print(f'[TRTR] Region: {args.region}' if args.model == 'trtr' else f'[TSTR] Region: {args.region}')

    # train-real-test-real 
    # build cache
    if args.force_rebuild:
        path = _cache_path(args.region)
        if os.path.exists(path):
            os.remove(path)
            print(f'[TRTR] Removed cache for region={args.region}')
    
    # load cache
    cache = load_cache(args.region)
    if cache is None:
        cache = trtr(args.dataset_dir, args.region, args.n_jobs)
        print(f'[TRTR] Built cache for region={args.region}')
    else:
        print(f'[TRTR] Loaded cache for region={args.region}')

    # save trtr results
    if args.model == 'trtr':
        result = {'model': 'trtr',
                  'region': args.region,
                  'n_train_real': cache['n_train'],
                  'n_synthetic': cache['n_train'],
                  'top_20_features': cache['top_20_features_sanitized'],
                  'metrics': cache['trtr_metrics'],
                  'feature_overlap': None,
                  'top_20_synth_features': None,
                  'trtr_metrics': cache['trtr_metrics']}
        save_result(result, args.model, args.region)
        save_summary(result)
        return

    # train-synthetic-test-real
    n_synthetic = args.n_synthetic if args.n_synthetic is not None else cache['n_train']
    result = tstr(cache, args.model, args.region, n_synthetic, args.n_jobs, device)
    
    # save tstr results
    save_result(result, args.model, args.region)
    save_summary(result)

if __name__ == '__main__':
    main()
