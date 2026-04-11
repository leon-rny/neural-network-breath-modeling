import argparse
import json
import os
import pickle
import re

import numpy as np
import pandas as pd
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

from core.data import CLASSES, CLASS_TO_IDX, BreathDataset, load_dataset, split_dataset

# hardcoded nose hyperparams from paper
_NOSE_PARAMS = dict(n_estimators=300, max_depth=6, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,)

# cli
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='TRTR pipeline ablation')
    p.add_argument('--region', required=True, choices=['mouth', 'nose'])
    p.add_argument('--pipeline', required=True, choices=['original', 'shap_fix', 'split_fix', 'full'], help='Which pipeline variant to run')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--dataset_dir', default='dataset')
    p.add_argument('--n_jobs', type=int, default=4)
    p.add_argument('--force_rebuild', action='store_true')
    return p.parse_args()

# utils
def df_to_df_long(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    hum_all  = np.stack(df['humidity'].values)
    temp_all = np.stack(df['temperature'].values)
    n, T = hum_all.shape
    ids   = np.repeat(np.arange(n), T)
    times = np.tile(np.arange(T), n)
    df_long = pd.DataFrame({'id': ids, 'time': times,
                            'Humidity': hum_all.ravel(),
                            'Temperature': temp_all.ravel()})
    labels = np.array([CLASS_TO_IDX[c] for c in df['class']])
    y = pd.Series(labels, index=np.arange(n), name='target')
    return df_long, y

def extract_fixed_features(df_long: pd.DataFrame, top_features_raw: list[str], n_jobs: int = 4) -> pd.DataFrame:
    kind_to_fc = from_columns(top_features_raw)
    X = extract_features(df_long, column_id='id', column_sort='time',
                         kind_to_fc_parameters=kind_to_fc, n_jobs=n_jobs)
    impute(X)
    missing = [c for c in top_features_raw if c not in X.columns]
    for c in missing:
        X[c] = 0.0
    return X[top_features_raw]

def train_stacking_classifier(X_train: np.ndarray, y_train: np.ndarray, seed: int) -> StackingClassifier:
    min_class = int(np.bincount(y_train).min())
    k_neighbors = min(5, min_class - 1)
    if k_neighbors < 1:
        X_res, y_res = X_train, y_train
    else:
        smote = SMOTE(random_state=seed, k_neighbors=k_neighbors)
        X_res, y_res = smote.fit_resample(X_train, y_train)
    xgb_clf  = XGBClassifier(eval_metric='mlogloss', random_state=seed, max_depth=4,
                              reg_alpha=0.5, reg_lambda=1.0, subsample=0.8,
                              colsample_bytree=0.8, n_estimators=300)
    cat_clf  = CatBoostClassifier(logging_level='Silent', random_state=seed, iterations=300,
                                   depth=4, l2_leaf_reg=5.0, random_strength=2.0,
                                   bagging_temperature=2.0, od_type='Iter', od_wait=20,
                                   allow_writing_files=False)
    meta_clf = RandomForestClassifier(n_estimators=150, max_depth=3, min_samples_leaf=5,
                                       min_samples_split=10, random_state=seed)
    stacker  = StackingClassifier(
        estimators=[('xgb', xgb_clf), ('cat', cat_clf)],
        final_estimator=meta_clf, passthrough=True,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed), n_jobs=-1)
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
            'per_class_f1': {cls: float(report.get(str(i), {}).get('f1-score', float('nan')))
                             for i, cls in enumerate(CLASSES)}}

def save_result(result: dict, run_id: str) -> None:
    os.makedirs('results/trtr_ablation', exist_ok=True)
    path = f'results/trtr_ablation/{run_id}_trtr.json'
    def _json_safe(obj):
        if isinstance(obj, float) and np.isnan(obj): return None
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list): return [_json_safe(v) for v in obj]
        return obj
    with open(path, 'w') as f:
        json.dump(_json_safe(result), f, indent=2)

def save_summary(result: dict) -> None:
    csv_path = 'results/trtr_ablation_summary.csv'
    m = result['metrics']
    new_row = {'pipeline': result['pipeline'],
               'region': result['region'],
               'seed': result['seed'],
               'accuracy': m['accuracy'],
               'f1_weighted': m['f1_weighted'],
               'roc_auc_ovr': m['roc_auc_ovr'],
               'log_loss': m['log_loss'],
               'f1_bradypnea': m['per_class_f1']['bradypnea'],
               'f1_eupnea': m['per_class_f1']['eupnea'],
               'f1_tachypnea': m['per_class_f1']['tachypnea'],
               'n_train': result['n_train']}
    df_row = pd.DataFrame([new_row])
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        df_new = (pd.concat([df_old, df_row], ignore_index=True)
                  .drop_duplicates(subset=['pipeline', 'region', 'seed'], keep='last'))
    else:
        df_new = df_row
    df_new.to_csv(csv_path, index=False)

# cache
def _cache_path(region: str, seed: int, pipeline: str) -> str:
    return f'results/trtr_ablation/{region}_s{seed}_{pipeline}_checkpoint.pkl'

def load_cache(region: str, seed: int, pipeline: str) -> dict | None:
    path = _cache_path(region, seed, pipeline)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

# trtr with ablation
def trtr(dataset_dir: str, region: str, n_jobs: int, seed: int, pipeline: str) -> dict:
    df = load_dataset(dataset_dir)
    df = df[df['region'] == region].reset_index(drop=True)

    # outer split
    if pipeline in ('original', 'shap_fix'):
        # 80/20 train/test split
        labels = np.array([CLASS_TO_IDX[c] for c in df['class']])
        idx = np.arange(len(df))
        idx_train, idx_test = train_test_split(idx, test_size=0.2, stratify=labels, random_state=seed)
        df_train = df.iloc[idx_train].reset_index(drop=True)
        df_test  = df.iloc[idx_test].reset_index(drop=True)
    else:
        # 80/10/10 train/val/test
        df_train, _, df_test = split_dataset(df, random_state=seed)

    stats = BreathDataset(df_train).stats

    # Feature extraction train only
    df_long_train, y_train = df_to_df_long(df_train)
    X_train_raw = extract_relevant_features(df_long_train, y_train,
                                            column_id='id', column_sort='time', n_jobs=n_jobs)
    impute(X_train_raw)

    raw_to_san = {col: re.sub(r'[^\w]', '_', col) for col in X_train_raw.columns}
    san_to_raw = {v: k for k, v in raw_to_san.items()}
    X_full_san = X_train_raw.copy()
    X_full_san.columns = [raw_to_san[c] for c in X_full_san.columns]

    # LightGBM for SHAP feature ranking
    X_tr, X_val_lgbm, y_tr, y_val_lgbm = train_test_split(
        X_full_san, y_train, test_size=0.2, stratify=y_train, random_state=seed)

    use_hardcoded_nose = (pipeline in ('original', 'shap_fix', 'split_fix') and region == 'nose')

    if use_hardcoded_nose:
        # Replicate: fixed hyperparams, no early stopping, fit on full train
        best_lgbm = LGBMClassifier(**_NOSE_PARAMS, random_state=seed, verbose=-1)
        best_lgbm.fit(X_full_san.values, y_train.values)
    else:
        # GridSearch + early stopping
        param_grid = {'max_depth': [4, 6], 'reg_alpha': [0.1, 1.0],
                      'reg_lambda': [0.5, 1.0], 'colsample_bytree': [0.8, 1.0]}
        base_lgbm = LGBMClassifier(n_estimators=1000, learning_rate=0.05, random_state=seed, verbose=-1)
        gs = GridSearchCV(base_lgbm, param_grid, cv=StratifiedKFold(3),
                          scoring='accuracy', n_jobs=-1, verbose=0)
        gs.fit(X_tr, y_tr)
        best_lgbm = LGBMClassifier(**gs.best_params_, n_estimators=1000,
                                    learning_rate=0.05, random_state=seed, verbose=-1)
        best_lgbm.fit(X_tr.values, y_tr.values,
                      eval_set=[(X_val_lgbm.values, y_val_lgbm.values)],
                      eval_metric='multi_logloss',
                      callbacks=[early_stopping(50, verbose=False), log_evaluation(0)])

    # Test features + SHAP
    df_long_test, y_test = df_to_df_long(df_test)
    explainer = shap.TreeExplainer(best_lgbm)

    if pipeline == 'original':
        # SHAP on test set
        X_test_raw_all = extract_fixed_features(df_long_test, list(X_train_raw.columns), n_jobs)
        X_test_san_all = X_test_raw_all.copy()
        X_test_san_all.columns = [re.sub(r'[^\w]', '_', c) for c in X_test_san_all.columns]
        X_test_san_all = X_test_san_all.reindex(columns=X_full_san.columns, fill_value=0.0)
        shap_values = explainer.shap_values(X_test_san_all.values)
    else:
        # SHAP on train
        shap_values = explainer.shap_values(X_full_san.values)

    mean_abs = np.abs(shap_values).mean(axis=0).mean(axis=1)
    top_idx   = np.argsort(mean_abs)[::-1][:20]
    top_20_san = X_full_san.columns.to_numpy()[top_idx].tolist()
    top_20_raw = [san_to_raw[s] for s in top_20_san]
    X_train_top = X_full_san[top_20_san].values

    if pipeline == 'original':
        X_test_top = X_test_san_all[top_20_san].values
    else:
        X_test_raw_top = extract_fixed_features(df_long_test, top_20_raw, n_jobs)
        X_test_san = X_test_raw_top.copy()
        X_test_san.columns = [re.sub(r'[^\w]', '_', c) for c in X_test_san.columns]
        X_test_top = X_test_san[top_20_san].values

    clf = train_stacking_classifier(X_train_top, y_train.values, seed)
    metrics = evaluate_classifier(clf, X_test_top, y_test.values)

    cache = {'top_20_features_raw': top_20_raw,
             'top_20_features_sanitized': top_20_san,
             'X_train_top': X_train_top,
             'X_test_top': X_test_top,
             'y_train': y_train.values,
             'y_test': y_test.values,
             'stats': stats,
             'n_train': len(df_train),
             'metrics': metrics,
             'seed': seed,
             'pipeline': pipeline}
    os.makedirs('results/trtr_ablation', exist_ok=True)
    with open(_cache_path(region, seed, pipeline), 'wb') as f:
        pickle.dump(cache, f)
    return cache

# main
def main():
    args = parse_args()
    np.random.seed(args.seed)

    run_id = f'{args.region}_s{args.seed}_{args.pipeline}'
    print(f'[TRTR-ABLATION] region={args.region}, pipeline={args.pipeline}, seed={args.seed}')

    if args.force_rebuild:
        path = _cache_path(args.region, args.seed, args.pipeline)
        if os.path.exists(path):
            os.remove(path)
            print(f'[TRTR-ABLATION] Removed cache: {path}')

    cache = load_cache(args.region, args.seed, args.pipeline)
    if cache is None:
        cache = trtr(args.dataset_dir, args.region, args.n_jobs, args.seed, args.pipeline)
        print(f'[TRTR-ABLATION] Built cache: {run_id}')
    else:
        print(f'[TRTR-ABLATION] Loaded cache: {run_id}')

    result = {'pipeline': args.pipeline,
              'region': args.region,
              'seed': args.seed,
              'n_train': cache['n_train'],
              'top_20_features': cache['top_20_features_sanitized'],
              'metrics': cache['metrics']}
    save_result(result, run_id)
    save_summary(result)
    print(f'[TRTR-ABLATION] Done. Accuracy={cache["metrics"]["accuracy"]:.3f}')

if __name__ == '__main__':
    main()
