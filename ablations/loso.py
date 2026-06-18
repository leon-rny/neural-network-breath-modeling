import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

COMMITTED_CONFIG = "a0.05_n10"
COMMITTED_BETA_MAX = 0.01
DEFAULT_SUMMARY = "results/ablation_cvae_jittering/summary.csv"
LOSO_DIR = "results/loso"


def build_shortlist(summary_path: str, region: str, k: int = 3) -> pd.DataFrame:
    """Rank (config, beta_max) candidates in `summary_path` for `region` by mean accuracy,
    keep the top-k distinct tuples (diversity guard), and force-include the committed config.
    Raises if the committed config is absent from the candidate pool (catches a wrong summary)."""
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"summary not found: {summary_path} (did the k-fold grid finish + aggregate?)")
    df = pd.read_csv(summary_path)
    for col in ("config", "beta_max", "region", "accuracy"):
        if col not in df.columns:
            raise ValueError(f"summary {summary_path} is missing column {col!r}; columns={list(df.columns)}")
    df = df[df["region"] == region]
    if df.empty:
        raise ValueError(f"no rows for region={region!r} in {summary_path}")

    # one row per (config, beta_max): mean accuracy over seeds x folds. groupby dedupes by identity.
    grp = (df.groupby(["config", "beta_max"], as_index=False)["accuracy"]
             .mean().sort_values("accuracy", ascending=False).reset_index(drop=True))

    # committed config must be in the candidate POOL (else we're reading the wrong summary)
    in_pool = ((grp["config"] == COMMITTED_CONFIG) & np.isclose(grp["beta_max"], COMMITTED_BETA_MAX)).any()
    if not in_pool:
        raise ValueError(
            f"committed config ({COMMITTED_CONFIG}, beta_max={COMMITTED_BETA_MAX}) absent from candidate pool "
            f"in {summary_path}. You are likely reading the wrong summary — the committed model was selected "
            f"from the jitter x beta grid (results/ablation_cvae_jittering/summary.csv).")

    short = grp.head(k).copy()
    # force-include the committed config so the actual model always gets a LOSO number
    committed_mask = (grp["config"] == COMMITTED_CONFIG) & np.isclose(grp["beta_max"], COMMITTED_BETA_MAX)
    if not ((short["config"] == COMMITTED_CONFIG) & np.isclose(short["beta_max"], COMMITTED_BETA_MAX)).any():
        short = pd.concat([short, grp[committed_mask]], ignore_index=True)
    short = short.drop_duplicates(subset=["config", "beta_max"]).reset_index(drop=True)
    return short


def _outer_fold_from_tag(loso_tag: str) -> int | None:
    """Recover the 1-indexed outer test fold t from a Stage-A `loso_tag` of the form `_nested_x{t0}`
    (t0 is the 0-indexed excluded subject). Returns None if the tag has no exclude (not a Stage-A row)."""
    m = re.search(r"_x(\d+)(?:-|$)", loso_tag or "")
    return int(m.group(1)) + 1 if m else None


def select_best(results_dir: str, region: str) -> pd.DataFrame:
    """For each outer test fold t, pick the (config, beta_max) with the best mean val accuracy
    across seeds, from the Stage-A `*_nested_x*_result.json` files in `results_dir`."""
    rows = []
    for path in glob.glob(os.path.join(results_dir, "*_nested_x*_result.json")):
        with open(path) as f:
            r = json.load(f)
        if r.get("region") != region:
            continue
        t = _outer_fold_from_tag(r.get("loso_tag", ""))
        if t is None:
            continue
        rows.append({"outer_fold": t, "config": r["config"], "beta_max": r["beta_max"],
                     "init_seed": r.get("init_seed"), "accuracy": r["accuracy"]})
    if not rows:
        raise ValueError(f"no Stage-A result JSONs (*_nested_x*) for region={region!r} in {results_dir}")
    df = pd.DataFrame(rows)
    # mean val accuracy per (outer_fold, config, beta_max) over seeds, then argmax per outer_fold
    agg = df.groupby(["outer_fold", "config", "beta_max"], as_index=False)["accuracy"].mean()
    best = agg.loc[agg.groupby("outer_fold")["accuracy"].idxmax()].sort_values("outer_fold").reset_index(drop=True)
    return best[["outer_fold", "config", "beta_max", "accuracy"]]


def _row_acc(r: dict, key: str):
    """Accuracy/f1 live flat in ablation result jsons but under `metrics` in core.tstr jsons."""
    if key in r and not isinstance(r.get(key), dict):
        return r.get(key)
    return (r.get("metrics") or {}).get(key)


def aggregate(ablation_dir: str = "results/ablation_cvae_jittering", trtr_dir: str = "results/trtr") -> pd.DataFrame:
    """Collect the Stage-B (final) LOSO results into one table: conv_baseline (`*_loso_nested_fold*`,
    i.e. nested final, no `_x` exclude) + the trtr baseline (`*_loso_nested_tstr.json`)."""
    rows = []
    for path in glob.glob(os.path.join(ablation_dir, "*_loso_nested_fold*_result.json")):
        r = json.load(open(path))
        rows.append({"model": "conv_baseline", "region": r.get("region"), "cv_mode": r.get("cv_mode"),
                     "fold": r.get("fold"), "init_seed": r.get("init_seed"), "config": r.get("config"),
                     "beta_max": r.get("beta_max"), "accuracy": _row_acc(r, "accuracy"), "f1_weighted": _row_acc(r, "f1_weighted")})
    for path in glob.glob(os.path.join(trtr_dir, "*_loso_nested_tstr.json")):
        r = json.load(open(path))
        rows.append({"model": "trtr", "region": r.get("region"), "cv_mode": r.get("cv_mode"),
                     "fold": r.get("fold"), "init_seed": r.get("init_seed"), "config": None,
                     "beta_max": None, "accuracy": _row_acc(r, "accuracy"), "f1_weighted": _row_acc(r, "f1_weighted")})
    if not rows:
        raise ValueError(f"no Stage-B LOSO result jsons found in {ablation_dir} / {trtr_dir}")
    return pd.DataFrame(rows).sort_values(["model", "region", "fold", "init_seed"]).reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Nested-LOSO shortlist / selection helper")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("shortlist", help="build the per-region candidate shortlist from the k-fold grid summary")
    ps.add_argument("--summary", default=DEFAULT_SUMMARY, help=f"k-fold grid summary csv (default {DEFAULT_SUMMARY})")
    ps.add_argument("--regions", default="mouth,nose")
    ps.add_argument("--k", type=int, default=3)

    pe = sub.add_parser("select", help="pick the best config per outer fold from Stage-A results")
    pe.add_argument("--results_dir", default="results/ablation_cvae_jittering")
    pe.add_argument("--regions", default="mouth,nose")

    pa = sub.add_parser("aggregate", help="consolidate Stage-B conv_baseline + trtr LOSO results")
    pa.add_argument("--ablation_dir", default="results/ablation_cvae_jittering")
    pa.add_argument("--trtr_dir", default="results/trtr")

    args = p.parse_args()
    os.makedirs(LOSO_DIR, exist_ok=True)

    if args.cmd == "shortlist":
        for region in args.regions.split(","):
            short = build_shortlist(args.summary, region, args.k)
            out = os.path.join(LOSO_DIR, f"shortlist_{region}.csv")
            short.to_csv(out, index=False)
            print(f"[LOSO] shortlist {region}: {len(short)} candidates -> {out}")
            print(short.to_string(index=False))
    elif args.cmd == "select":
        for region in args.regions.split(","):
            best = select_best(args.results_dir, region)
            out = os.path.join(LOSO_DIR, f"selected_{region}.csv")
            best.to_csv(out, index=False)
            print(f"[LOSO] selected {region}: {len(best)} folds -> {out}")
            print(best.to_string(index=False))
    elif args.cmd == "aggregate":
        table = aggregate(args.ablation_dir, args.trtr_dir)
        out = os.path.join(LOSO_DIR, "summary_loso.csv")
        table.to_csv(out, index=False)
        print(f"[LOSO] aggregated {len(table)} Stage-B rows -> {out}")
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
