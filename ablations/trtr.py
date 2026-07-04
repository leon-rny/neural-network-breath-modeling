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
from imblearn.pipeline import Pipeline as ImbPipeline
import shap
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from core.data import CLASSES, CLASS_TO_IDX, BreathDataset, load_dataset

# 1. original: pipeline on a single 80/10/10 split (no CV)
# 2. replication: paper's pipeline, with leakage in SHAP, tsfresh, SMOTE and hardcoded nose hyperparams
# 3. shap_fix: + SHAP computed on train only (was: on test)
# 4. lgbm_fix: + unified LightGBM tuning (was: hardcoded nose params)
# 5. tsfresh_fix: + tsfresh feature selection per fold (was: on full dataset)
# 6. smote_fix: + SMOTE inside the stacker's CV pipeline (was: global)
_PIPELINES = ("replication", "shap_fix", "lgbm_fix", "tsfresh_fix", "smote_fix")

# cli
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="TRTR pipeline ablation")
    p.add_argument("--region", choices=["mouth", "nose"], default=None)  # required except in --aggregate
    p.add_argument("--pipeline", choices=list(_PIPELINES), default=None)
    p.add_argument("--init_seed", type=int, default=42)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--single_split", action="store_true")
    p.add_argument("--dataset_dir", default="dataset")
    p.add_argument("--n_jobs", type=int, default=4)
    p.add_argument("--force_rebuild", action="store_true")
    p.add_argument("--no_summary", action="store_true")
    p.add_argument("--aggregate", action="store_true")
    return p.parse_args()

# utils
def df_to_df_long(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Reshape a trial dataframe into the tsfresh long format and class labels.

    :param df: dataframe with per-trial 'humidity'/'temperature' arrays and a 'class' column.
    :return: (df_long, y) where df_long has columns id/time/Humidity/Temperature (one row per
        sample) and y is an int class-index Series indexed by trial id.
    """
    hum_all  = np.stack(df["humidity"].values)
    temp_all = np.stack(df["temperature"].values)
    n, T = hum_all.shape
    ids   = np.repeat(np.arange(n), T)
    times = np.tile(np.arange(T), n)
    df_long = pd.DataFrame({"id": ids, "time": times,
                            "Humidity": hum_all.ravel(),
                            "Temperature": temp_all.ravel()})
    labels = np.array([CLASS_TO_IDX[c] for c in df["class"]])
    y = pd.Series(labels, index=np.arange(n), name="target")
    return df_long, y

def extract_fixed_features(df_long: pd.DataFrame, top_features_raw: list[str], n_jobs: int = 4) -> pd.DataFrame:
    """Extract a fixed set of tsfresh features from long-format data.

    :param df_long: long-format data with id/time/channel columns.
    :param top_features_raw: raw tsfresh feature names to compute (order preserved).
    :param n_jobs: parallel workers for tsfresh.
    :return: DataFrame with exactly top_features_raw as columns, any feature tsfresh fails
        to produce is filled with 0.0.
    """
    kind_to_fc = from_columns(top_features_raw)
    X = extract_features(df_long, column_id="id", column_sort="time", kind_to_fc_parameters=kind_to_fc, n_jobs=n_jobs)
    impute(X)
    missing = [c for c in top_features_raw if c not in X.columns]
    for c in missing:
        X[c] = 0.0
    return X[top_features_raw]

def train_stacking_classifier(X_train: np.ndarray, y_train: np.ndarray, init_seed: int, fix_smote: bool = False, n_jobs: int = -1) -> StackingClassifier:
    """Fit a SMOTE + XGB/CatBoost stacking classifier with a random-forest meta-learner.

    :param X_train: feature matrix.
    :param y_train: integer class labels.
    :param init_seed: random seed for all estimators and SMOTE.
    :param fix_smote: if True, apply SMOTE inside each base estimator's CV pipeline (no leakage),
        if False, oversample globally before stacking (leaks across the stacker's internal folds).
    :param n_jobs: parallel workers for the stacker's cross-validation.
    :return: the fitted StackingClassifier.
    """
    min_class = int(np.bincount(y_train).min())
    k_neighbors = min(5, min_class - 1)

    xgb_kw = dict(eval_metric="mlogloss", random_state=init_seed, max_depth=4,
                  reg_alpha=0.5, reg_lambda=1.0, subsample=0.8,
                  colsample_bytree=0.8, n_estimators=300)
    cat_kw = dict(logging_level="Silent", random_state=init_seed, iterations=300,
                  depth=4, l2_leaf_reg=5.0, random_strength=2.0,
                  bagging_temperature=2.0, od_type="Iter", od_wait=20,
                  allow_writing_files=False)

    if fix_smote:
        # SMOTE inside each base estimator's pipeline: resampling happens inside the stacker's internal CV folds, no leakage into hold-outs.
        def make_base(clf):
            """Wrap a base estimator in a SMOTE pipeline, or return it unchanged if SMOTE is infeasible."""
            if k_neighbors < 1:
                return clf
            return ImbPipeline([("smote", SMOTE(random_state=init_seed, k_neighbors=k_neighbors)),
                                ("clf", clf)])
        xgb_clf = make_base(XGBClassifier(**xgb_kw))
        cat_clf = make_base(CatBoostClassifier(**cat_kw))
        X_fit, y_fit = X_train, y_train
    else:
        # SMOTE applied globally before stacking: leaks across the stacker's internal CV folds.
        if k_neighbors < 1:
            X_fit, y_fit = X_train, y_train
        else:
            smote = SMOTE(random_state=init_seed, k_neighbors=k_neighbors)
            X_fit, y_fit = smote.fit_resample(X_train, y_train)
        xgb_clf = XGBClassifier(**xgb_kw)
        cat_clf = CatBoostClassifier(**cat_kw)

    meta_clf = RandomForestClassifier(n_estimators=150, max_depth=3, min_samples_leaf=5,
                                       min_samples_split=10, random_state=init_seed)
    stacker  = StackingClassifier(
        estimators=[("xgb", xgb_clf), ("cat", cat_clf)],
        final_estimator=meta_clf, passthrough=True,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=init_seed), n_jobs=n_jobs)
    stacker.fit(X_fit, y_fit)
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
        roc_auc = float(roc_auc_score(y_test, y_prob, multi_class="ovr"))
    except Exception as e:
        print(f"ROC-AUC failed: {e}")
        roc_auc = float("nan")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return {"accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "roc_auc_ovr": roc_auc,
            "log_loss": float(log_loss(y_test, y_prob)),
            "per_class_f1": {cls: float(report.get(str(i), {}).get("f1-score", float("nan"))) for i, cls in enumerate(CLASSES)}}

def save_result(result: dict, run_id: str) -> None:
    """Write a result dict to results/ablation_trtr/<run_id>_trtr.json as JSON-safe values.

    :param result: the result dict (nan/numpy values are converted on write).
    :param run_id: artifact id used as the filename stem.
    :return: None.
    """
    os.makedirs("results/ablation_trtr", exist_ok=True)
    path = f"results/ablation_trtr/{run_id}_trtr.json"
    def _json_safe(obj):
        """Recursively convert nan and numpy scalars/arrays into JSON-serializable values."""
        if isinstance(obj, float) and np.isnan(obj):
            return None
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_json_safe(v) for v in obj]
        return obj
    with open(path, "w") as f:
        json.dump(_json_safe(result), f, indent=2)

def save_summary(result: dict) -> None:
    """Append a flattened result row to results/ablation_trtr/summary.csv, deduping by config keys.

    Reconciles the single_split column with any existing file and keeps the last row per unique
    (pipeline, region, init_seed, split_seed, single_split, fold, n_folds) combination.

    :param result: a result dict as produced by trtr.
    :return: None.
    """
    csv_path = "results/ablation_trtr/summary.csv"
    m = result["metrics"]
    new_row = {"pipeline": result["pipeline"],
               "region": result["region"],
               "init_seed": result["init_seed"],
               "split_seed": result["split_seed"],
               "single_split": bool(result.get("single_split", False)),
               "fold": result["fold"],
               "n_folds": result["n_folds"],
               "accuracy": m["accuracy"],
               "f1_weighted": m["f1_weighted"],
               "roc_auc_ovr": m["roc_auc_ovr"],
               "log_loss": m["log_loss"],
               "f1_bradypnea": m["per_class_f1"]["bradypnea"],
               "f1_eupnea": m["per_class_f1"]["eupnea"],
               "f1_tachypnea": m["per_class_f1"]["tachypnea"],
               "n_train": result["n_train"]}
    df_row = pd.DataFrame([new_row])
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        if "single_split" not in df_old.columns:
            df_old["single_split"] = False
        else:
            df_old["single_split"] = df_old["single_split"].fillna(False).astype(bool)
        df_new = (pd.concat([df_old, df_row], ignore_index=True)
                  .drop_duplicates(subset=["pipeline", "region", "init_seed", "split_seed", "single_split", "fold", "n_folds"], keep="last"))
    else:
        df_new = df_row
    df_new.to_csv(csv_path, index=False)

# cache
def _split_tag(single_split: bool, fold: int, n_folds: int) -> str:
    """Return the run-id split tag: 'single' for a single split, else 'fold{fold}of{n_folds}'."""
    return "single" if single_split else f"fold{fold}of{n_folds}"

def _cache_path(region: str, init_seed: int, split_seed: int, fold: int, n_folds: int, pipeline: str, single_split: bool = False) -> str:
    """Build the trtr ablation checkpoint cache path for a given configuration."""
    tag = _split_tag(single_split, fold, n_folds)
    return f"results/ablation_trtr/{region}_is{init_seed}_ss{split_seed}_{tag}_{pipeline}_checkpoint.pkl"

def load_cache(region: str, init_seed: int, split_seed: int, fold: int, n_folds: int, pipeline: str, single_split: bool = False) -> dict | None:
    """Load the cached trtr ablation checkpoint for a configuration, or None if it does not exist."""
    path = _cache_path(region, init_seed, split_seed, fold, n_folds, pipeline, single_split)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

# trtr with ablation
def trtr(dataset_dir: str, region: str, n_jobs: int, init_seed: int, split_seed: int, fold: int, n_folds: int, pipeline: str, single_split: bool = False) -> dict:
    """Run one train-real-test-real pipeline ablation and cache its artifacts.

    Toggles the leakage behaviours that distinguish the pipeline stages (SHAP on test,
    hardcoded nose LGBM hyperparams, tsfresh selection on the full dataset, SMOTE outside
    the stacker's CV), then loads/splits the region data, selects the top-20 SHAP features,
    trains the stacking classifier, evaluates on the real test set, pickles the cache to
    results/ablation_trtr, and returns it.

    :param dataset_dir: dataset root directory.
    :param region: 'mouth' or 'nose'.
    :param n_jobs: parallel workers for feature extraction and the classifier.
    :param init_seed: model/estimator seed.
    :param split_seed: data-split seed.
    :param fold: 1-indexed fold (ignored when single_split).
    :param n_folds: number of cv folds.
    :param pipeline: ablation stage name (one of _PIPELINES) selecting which leaks are present.
    :param single_split: use one legacy 80/10/10 split instead of stratified k-fold.
    :return: cache dict with keys top_20_features_raw, top_20_features_sanitized, X_train_top,
        X_test_top, y_train, y_test, stats, n_train, metrics, init_seed, split_seed, fold,
        n_folds, single_split, pipeline.
    """
    shap_on_test = (pipeline == "replication")
    nose_hardcoded = (pipeline in ("replication", "shap_fix") and region == "nose")
    tsfresh_on_full = (pipeline in ("replication", "shap_fix", "lgbm_fix"))
    smote_inside_cv = (pipeline == "smote_fix")

    df = load_dataset(dataset_dir)
    df = df[df["region"] == region].reset_index(drop=True)
    labels_full = np.array([CLASS_TO_IDX[c] for c in df["class"]])

    if single_split:
        # Legacy protocol: one 80/10/10 stratified train/val/test split
        idx_all = np.arange(len(df))
        idx_trainval, test_idx = train_test_split(idx_all, test_size=0.1, stratify=labels_full, random_state=split_seed)
        val_relative = 0.1 / (1 - 0.1)
        train_idx, _ = train_test_split(idx_trainval, test_size=val_relative,
                                        stratify=labels_full[idx_trainval], random_state=split_seed)
    else:
        # fold is 1-indexed at the API boundary, splits list is 0-indexed.
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=split_seed)
        splits = list(skf.split(df, labels_full))
        trainfull_idx, test_idx = splits[fold - 1]
        trainfull_pos = np.arange(len(trainfull_idx))
        train_pos, _ = train_test_split(trainfull_pos, test_size=0.15,
                                        stratify=labels_full[trainfull_idx], random_state=split_seed)
        train_idx = trainfull_idx[train_pos]

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test  = df.iloc[test_idx].reset_index(drop=True)
    stats = BreathDataset(df_train).stats

    # tsfresh feature selection
    if tsfresh_on_full:
        # Leaky: select features on the full pre-split dataset. Selection uses test-set labels -> information leaks into the feature set.
        df_long_full, y_full = df_to_df_long(df)
        X_full_raw = extract_relevant_features(df_long_full, y_full, column_id="id", column_sort="time", n_jobs=n_jobs)
        impute(X_full_raw)
        X_train_raw = X_full_raw.iloc[train_idx].reset_index(drop=True)
        y_train_values = labels_full[train_idx]
    else:
        # Correct: select features on the training partition only.
        df_long_train, y_train_series = df_to_df_long(df_train)
        X_train_raw = extract_relevant_features(df_long_train, y_train_series, column_id="id", column_sort="time", n_jobs=n_jobs)
        impute(X_train_raw)
        y_train_values = y_train_series.values

    raw_to_san = {col: re.sub(r"[^\w]", "_", col) for col in X_train_raw.columns}
    san_to_raw = {v: k for k, v in raw_to_san.items()}
    X_full_san = X_train_raw.copy()
    X_full_san.columns = [raw_to_san[c] for c in X_full_san.columns]

    # LightGBM for SHAP feature ranking. Inner train/val partition uses split_seed (data partition), LGBM init uses init_seed (model stochasticity).
    X_tr, X_val_lgbm, y_tr, y_val_lgbm = train_test_split(
        X_full_san, y_train_values, test_size=0.2, stratify=y_train_values, random_state=split_seed)

    if nose_hardcoded:
        # Replicate paper: fixed hyperparams, no early stopping, fit on full train.
        nose_params = dict(n_estimators=300, max_depth=6, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0)
        best_lgbm = LGBMClassifier(**nose_params, random_state=init_seed, verbose=-1)
        best_lgbm.fit(X_full_san.values, y_train_values)
    else:
        # Unified tuning: GridSearch + early stopping.
        param_grid = {"max_depth": [4, 6], "reg_alpha": [0.1, 1.0],
                      "reg_lambda": [0.5, 1.0], "colsample_bytree": [0.8, 1.0]}
        base_lgbm = LGBMClassifier(n_estimators=1000, learning_rate=0.05, random_state=init_seed, verbose=-1)
        gs = GridSearchCV(base_lgbm, param_grid, cv=StratifiedKFold(3),
                          scoring="accuracy", n_jobs=n_jobs, verbose=0)
        gs.fit(X_tr, y_tr)
        best_lgbm = LGBMClassifier(**gs.best_params_, n_estimators=1000, learning_rate=0.05, random_state=init_seed, verbose=-1)
        best_lgbm.fit(X_tr.values, y_tr,
                      eval_set=[(X_val_lgbm.values, y_val_lgbm)],
                      eval_metric="multi_logloss",
                      callbacks=[early_stopping(50, verbose=False), log_evaluation(0)])

    # Test features
    df_long_test, y_test = df_to_df_long(df_test)
    y_test_values = y_test.values

    if tsfresh_on_full:
        # Reuse the already-extracted full-dataset feature matrix, sliced for test rows.
        X_test_san_all = X_full_raw.iloc[test_idx].reset_index(drop=True).copy()
        X_test_san_all.columns = [re.sub(r"[^\w]", "_", c) for c in X_test_san_all.columns]
        X_test_san_all = X_test_san_all.reindex(columns=X_full_san.columns, fill_value=0.0)
    else:
        # Extract test features fresh, restricted to train-selected columns.
        X_test_raw_all = extract_fixed_features(df_long_test, list(X_train_raw.columns), n_jobs)
        X_test_san_all = X_test_raw_all.copy()
        X_test_san_all.columns = [re.sub(r"[^\w]", "_", c) for c in X_test_san_all.columns]
        X_test_san_all = X_test_san_all.reindex(columns=X_full_san.columns, fill_value=0.0)

    # SHAP feature ranking. Replication computes SHAP on the test set (information leak), every fix in the chain switches to SHAP on train.
    explainer = shap.TreeExplainer(best_lgbm)
    if shap_on_test:
        shap_values = explainer.shap_values(X_test_san_all.values)
    else:
        shap_values = explainer.shap_values(X_full_san.values)

    mean_abs = np.abs(shap_values).mean(axis=0).mean(axis=1)
    top_idx = np.argsort(mean_abs)[::-1][:20]
    top_20_san = X_full_san.columns.to_numpy()[top_idx].tolist()
    top_20_raw = [san_to_raw[s] for s in top_20_san]
    X_train_top = X_full_san[top_20_san].values
    X_test_top  = X_test_san_all[top_20_san].values

    clf = train_stacking_classifier(X_train_top, y_train_values, init_seed, fix_smote=smote_inside_cv, n_jobs=n_jobs)
    metrics = evaluate_classifier(clf, X_test_top, y_test_values)

    cache = {"top_20_features_raw": top_20_raw,
             "top_20_features_sanitized": top_20_san,
             "X_train_top": X_train_top,
             "X_test_top": X_test_top,
             "y_train": y_train_values,
             "y_test": y_test_values,
             "stats": stats,
             "n_train": len(df_train),
             "metrics": metrics,
             "init_seed": init_seed,
             "split_seed": split_seed,
             "fold": fold,
             "n_folds": n_folds,
             "single_split": single_split,
             "pipeline": pipeline}
    os.makedirs("results/ablation_trtr", exist_ok=True)
    with open(_cache_path(region, init_seed, split_seed, fold, n_folds, pipeline, single_split), "wb") as f:
        pickle.dump(cache, f)
    return cache

# main
def main():
    """Run one TRTR pipeline-ablation config (or aggregate-only summary merge) from CLI args."""
    args = parse_args()

    # aggregate mode
    if args.aggregate:
        import glob
        paths = sorted(glob.glob("results/ablation_trtr/*_trtr.json"))
        for path in paths:
            with open(path) as f:
                save_summary(json.load(f))
        print(f"[TRTR-ABLATION] aggregated {len(paths)} results into results/ablation_trtr/summary.csv")
        return
    if args.region is None or args.pipeline is None:
        raise SystemExit("[TRTR-ABLATION] --region and --pipeline are required (except with --aggregate)")

    np.random.seed(args.init_seed)

    split_tag = _split_tag(args.single_split, args.fold, args.n_folds)
    run_id = f"{args.region}_is{args.init_seed}_ss{args.split_seed}_{split_tag}_{args.pipeline}"
    split_desc = "single 80/10/10" if args.single_split else f"fold {args.fold}/{args.n_folds}"
    print(f"[TRTR-ABLATION] region={args.region} pipeline={args.pipeline} init_seed={args.init_seed} split_seed={args.split_seed} split={split_desc}")

    if args.force_rebuild:
        path = _cache_path(args.region, args.init_seed, args.split_seed, args.fold, args.n_folds, args.pipeline, args.single_split)
        if os.path.exists(path):
            os.remove(path)
            print(f"[TRTR-ABLATION] Removed cache: {path}")

    cache = load_cache(args.region, args.init_seed, args.split_seed, args.fold, args.n_folds, args.pipeline, args.single_split)
    if cache is None:
        cache = trtr(args.dataset_dir, args.region, args.n_jobs, args.init_seed, args.split_seed, args.fold, args.n_folds, args.pipeline, args.single_split)
        print(f"[TRTR-ABLATION] Built cache: {run_id}")
    else:
        print(f"[TRTR-ABLATION] Loaded cache: {run_id}")

    result = {"pipeline": args.pipeline,
              "region": args.region,
              "init_seed": args.init_seed,
              "split_seed": args.split_seed,
              "single_split": args.single_split,
              "fold": args.fold,
              "n_folds": args.n_folds,
              "n_train": cache["n_train"],
              "top_20_features": cache["top_20_features_sanitized"],
              "metrics": cache["metrics"]}
    save_result(result, run_id)
    if not args.no_summary:
        save_summary(result)
    print(f"[TRTR-ABLATION] Done. Accuracy={cache['metrics']['accuracy']:.3f}")

if __name__ == "__main__":
    main()
