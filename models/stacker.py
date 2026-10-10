"""Reference classifier of the thesis: tsfresh features, top-20 by SHAP of a tuned LightGBM, stacking of SMOTE + XGBoost
and SMOTE + CatBoost with a random-forest meta-learner; real trials only."""

import re

import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from tsfresh import extract_features, select_features
from tsfresh.feature_extraction.settings import from_columns
from tsfresh.utilities.dataframe_functions import (
    get_range_values_per_column,
    impute_dataframe_range,
)
from xgboost import XGBClassifier

from core.data import CLASS_TO_IDX

SPLIT_SEED = 42  # internal hold-out and grid-search folds

def baseline_corrected(df: pd.DataFrame, nb: int = 5) -> pd.DataFrame:
    df = df.copy()
    for col in ('humidity', 'temperature'):
        df[col] = df[col].apply(lambda a: np.asarray(a, dtype=float) - np.asarray(a, dtype=float)[:nb].mean())
    return df

def df_to_df_long(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    # tsfresh long format (id, time, Humidity, Temperature; one row per sample) and the class indices by trial id
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

def extract_fixed_features(df_long: pd.DataFrame, top_features_raw: list[str], n_jobs: int, imputation: dict) -> pd.DataFrame:
    # the given raw tsfresh features (order preserved), imputed with the training values ('max', 'min', 'median')
    kind_to_fc = from_columns(top_features_raw)
    X = extract_features(df_long, column_id='id', column_sort='time', kind_to_fc_parameters=kind_to_fc, n_jobs=n_jobs)
    for col in top_features_raw:
        if col not in X:
            X[col] = np.nan
    impute_dataframe_range(X, imputation['max'], imputation['min'], imputation['median'])
    return X[top_features_raw]

class TrainingFeatureSelector(TransformerMixin, BaseEstimator):
    """Imputation values and tsfresh relevance selection learned from training rows only."""

    def __init__(self, n_jobs: int = 1) -> None:
        self.n_jobs = n_jobs

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'TrainingFeatureSelector':
        self.max_, self.min_, self.median_ = get_range_values_per_column(X)
        clean = impute_dataframe_range(X.copy(), self.max_, self.min_, self.median_)
        self.columns_ = select_features(clean, y, n_jobs=self.n_jobs).columns.tolist()
        if not self.columns_:
            raise ValueError('No relevant features found in the inner training fold')
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        clean = impute_dataframe_range(X.copy(), self.max_, self.min_, self.median_)
        return clean[self.columns_]

class FoldSMOTE(SMOTE):
    """SMOTE that falls back to random oversampling when a class has a single training sample in a fold."""

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _, counts = np.unique(y, return_counts=True)
        if len(counts) < 2:
            raise ValueError('Classification needs at least two training classes')
        if counts.min() < 2:
            return RandomOverSampler(random_state=self.random_state).fit_resample(X, y)
        sampler = SMOTE(random_state=self.random_state, k_neighbors=min(5, int(counts.min()) - 1))
        return sampler.fit_resample(X, y)

def train_stacking_classifier(X_train: np.ndarray, y_train: np.ndarray, init_seed: int, n_jobs: int = -1) -> StackingClassifier:
    """SMOTE + XGBoost / CatBoost stacking with a random-forest meta-learner (passthrough).

    :param init_seed: seed of all estimators and SMOTE
    :param n_jobs: parallel workers of the stacker's cross-validation
    :return: the fitted classifier
    """
    def make_base(clf: BaseEstimator) -> ImbPipeline:
        return ImbPipeline([('smote', FoldSMOTE(random_state=init_seed)),
                            ('clf', clf)])

    min_class = int(np.bincount(y_train).min())
    if min_class < 2:
        raise ValueError('Stacking cross-validation needs at least two samples per class')
    cv = StratifiedKFold(n_splits=min(5, min_class), shuffle=True, random_state=init_seed)
    xgb_clf = make_base(XGBClassifier(n_jobs=1, eval_metric='mlogloss', random_state=init_seed, max_depth=4, reg_alpha=0.5, reg_lambda=1.0, subsample=0.8,
                                      colsample_bytree=0.8, n_estimators=300))
    cat_clf = make_base(CatBoostClassifier(thread_count=1, logging_level='Silent', random_state=init_seed, iterations=300, depth=4, l2_leaf_reg=5.0,
                                           random_strength=2.0, bagging_temperature=2.0, od_type='Iter', od_wait=20, allow_writing_files=False))
    meta_clf = RandomForestClassifier(n_jobs=1, n_estimators=150, max_depth=3, min_samples_leaf=5, min_samples_split=10, random_state=init_seed)
    stacker = StackingClassifier(estimators=[('xgb', xgb_clf), ('cat', cat_clf)], final_estimator=meta_clf, passthrough=True, cv=cv, n_jobs=n_jobs)
    stacker.fit(X_train, y_train)
    return stacker

def fit_predict(df_train: pd.DataFrame, df_test: pd.DataFrame, init_seed: int, n_jobs: int = 1) -> np.ndarray:
    """Train the stacker on the training trials and predict the test trials.

    :param df_train: training trials (absolute signals, baseline-corrected here)
    :param init_seed: model / estimator seed
    :param n_jobs: parallel workers of the feature extraction, grid search and stacker
    :return: class probabilities of the test trials
    """
    df_train, df_test = baseline_corrected(df_train), baseline_corrected(df_test)
    df_long_train, y_train = df_to_df_long(df_train)
    # feature extraction
    X_train_raw = extract_features(df_long_train, column_id='id', column_sort='time', n_jobs=n_jobs)
    raw_to_san = {col: re.sub(r'[^\w]', '_', col) for col in X_train_raw.columns}
    san_to_raw = {v: k for k, v in raw_to_san.items()}
    X_full_san = X_train_raw.copy()
    X_full_san.columns = [raw_to_san[c] for c in X_full_san.columns]
    # lgbm grid search (3-fold) with training-only relevance filter, early-stopped on a 20 % hold-out
    X_tr, X_val, y_tr, y_val = train_test_split(X_full_san, y_train, test_size=0.2, stratify=y_train, random_state=SPLIT_SEED)
    param_grid = {'max_depth': [4, 6],
                  'reg_alpha': [0.1, 1.0],
                  'reg_lambda': [0.5, 1.0],
                  'colsample_bytree': [0.8, 1.0]}
    base_lgbm = LGBMClassifier(n_jobs=1, n_estimators=1000, learning_rate=0.05, random_state=init_seed, verbose=-1)
    pipeline = ImbPipeline([('features', TrainingFeatureSelector(n_jobs=n_jobs)), ('model', base_lgbm)])
    grid = {f'model__{key}': value for key, value in param_grid.items()}
    gs = GridSearchCV(pipeline, grid, cv=StratifiedKFold(3, shuffle=True, random_state=SPLIT_SEED), scoring='accuracy', n_jobs=n_jobs, verbose=0)
    gs.fit(X_tr, y_tr)
    selector = gs.best_estimator_.named_steps['features']
    X_tr, X_val = selector.transform(X_tr), selector.transform(X_val)
    X_full_san = selector.transform(X_full_san)
    imputation = {key: {san_to_raw[col]: value for col, value in values.items()}
                  for key, values in (('max', selector.max_), ('min', selector.min_), ('median', selector.median_))}
    best_params = {key.removeprefix('model__'): value for key, value in gs.best_params_.items()}
    best_lgbm = LGBMClassifier(**best_params, n_jobs=1, n_estimators=1000, learning_rate=0.05, random_state=init_seed, verbose=-1)
    best_lgbm.fit(X_tr.values, y_tr.values, eval_set=[(X_val.values, y_val.values)], eval_metric='multi_logloss',
                  callbacks=[early_stopping(50, verbose=False), log_evaluation(0)])
    # top-20 features by mean |SHAP|
    explainer = shap.TreeExplainer(best_lgbm)
    shap_values = explainer.shap_values(X_full_san.values)
    mean_abs = np.abs(shap_values).mean(axis=0).mean(axis=1)
    top_idx = np.argsort(mean_abs)[::-1][:20]
    top_20_san = X_full_san.columns.to_numpy()[top_idx].tolist()
    top_20_raw = [san_to_raw[s] for s in top_20_san]
    X_train_top = X_full_san[top_20_san].values
    # test trials: the same 20 features with the training imputation
    df_long_test, _ = df_to_df_long(df_test)
    X_test_san = extract_fixed_features(df_long_test, top_20_raw, n_jobs, imputation).copy()
    X_test_san.columns = [re.sub(r'[^\w]', '_', col) for col in X_test_san.columns]
    X_test_top = X_test_san[top_20_san].values
    # stacker
    clf = train_stacking_classifier(X_train_top, y_train.values, init_seed, n_jobs=n_jobs)
    return clf.predict_proba(X_test_top)
