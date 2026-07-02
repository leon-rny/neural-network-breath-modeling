import argparse
import glob
import json
import os
import random
import re

import numpy as np
import pandas as pd

from core.data import load_dataset, n_loso_folds, subset_tag

# --- shared config (inlined from the former four-objective sweep) ---
# objective -> (mode, cv_mode, part_dropout)
OBJECTIVES = {
    'tstr': ('tstr', 'kfold', 0.0),
    'loso': ('tstr', 'loso', 0.1),
    'tstrp': ('tstr_plus', 'kfold', 0.0),
    'losop': ('tstr_plus', 'loso', 0.1),
}
MODELS = ['tpinn']
MODEL0 = {'cvae_part': 'c', 'tpinn': 't'}
REGIONS = ['mouth', 'nose']
# committed anchors (config 0)
ANCHOR = {
    'cvae_part': dict(ld=16, ed=8, ped=8, fb=0.0, bm=0.01, alpha=0.05, ncop=10),
    'tpinn': dict(ld=16, ed=8, ped=8, fb=0.0, bm=0.01, alpha=0.05, ncop=10),
}
SPACE = {
    'cvae_part': dict(ld=[8, 16, 32], ed=[4, 8, 16], ped=[4, 8, 16], fb=[0.0, 0.1, 0.5],
                      bm=[0.003, 0.01, 0.03, 0.1], alpha=[0.0, 0.05, 0.1], ncop=[1, 5, 10, 20]),
    'tpinn': dict(ld=[8, 16, 32], ed=[8], ped=[8], fb=[0.0, 0.5],
                  bm=[0.003, 0.01, 0.03], alpha=[0.0, 0.05, 0.1], ncop=[1, 5, 10, 20]),
}
AUGR = [0.5, 1.0, 2.0, 3.0]
N_CONFIGS = 16


def sample_configs(model, objective, mi, oi):
    """Deterministic per (model, objective): config 0 = anchor, rest = seeded random samples from SPACE."""
    rng = random.Random(1000 * mi + oi)
    sp = SPACE[model]
    plus = objective in ('tstrp', 'losop')
    cfgs = []
    anchor = dict(ANCHOR[model])
    anchor['augr'] = 1.0
    cfgs.append(anchor)
    seen = {tuple(sorted(anchor.items()))}
    tries = 0
    while len(cfgs) < N_CONFIGS and tries < 5000:
        tries += 1
        c = {k: rng.choice(v) for k, v in sp.items()}
        c['augr'] = rng.choice(AUGR) if plus else 1.0
        if c['alpha'] == 0.0:
            c['ncop'] = 1  # n_copies only matters when alpha>0; canonicalize to avoid dup configs
        key = tuple(sorted(c.items()))
        if key in seen:
            continue
        seen.add(key)
        cfgs.append(c)
    return cfgs

# ---------------------------------------------------------------------------
# protocol constants
# ---------------------------------------------------------------------------
PILOT_SEEDS = [0, 1, 7, 42, 123]   # full seed grid for the noise-floor pilot
SELECT_SEEDS = [0, 1, 7]           # inner-selection seeds (inner folds already average a lot)
FULL_SEEDS = [0, 1, 7, 42, 123]    # Stage-B final seed grid
N_DRAWS_PILOT = 5                  # synthetic resample draws per trained model (cheap, no retrain)
N_DRAWS_SEL = 3
N_DRAWS_FINAL = 5
N_PILOT_CONFIGS = 6                # anchor + 5 diverse, to gauge config sensitivity
N_SEARCH_CONFIGS = 12              # deterministic config shortlist enumerated by the nested search
HP_KEYS = ('ld', 'ed', 'ped', 'fb', 'bm', 'alpha', 'ncop', 'augr')

# physics flags for the tpinn anchor/search (mirrors experiments/41_tune_four_objectives.sh)
TPINN_PHYS = dict(phys_residual=True, phys_prep='stdscale')


def _phys_flags(model: str) -> dict:
    """Return the physics build-id flags for `model` ({} for cvae_part, residual+stdscale for tpinn)."""
    return dict(TPINN_PHYS) if model == 'tpinn' else dict(phys_residual=False, phys_prep='peakscale')


def _complexity(cfg: dict) -> tuple:
    """Sort key for the 1-SE rule: prefer the simplest (smallest-capacity, most-regularised) config.
    Lower is simpler: small latent+embed capacity, then larger free_bits, then lower cfg_id."""
    return (cfg['ld'] + cfg['ed'] + cfg['ped'], -float(cfg['fb']), int(cfg.get('cfg_id', 0)))


def _pilot_configs(model: str, mi: int) -> list[dict]:
    """Anchor (cfg 0) + a diverse spread of configs (with augr) to estimate config sensitivity."""
    oi = list(OBJECTIVES).index('losop')
    return sample_configs(model, 'losop', mi, oi)[:N_PILOT_CONFIGS]


def _search_configs(model: str, cv: str, mi: int) -> list[dict]:
    """Deterministic config shortlist for the nested search (uses the augr-bearing '+' sampler)."""
    objective = 'losop' if cv == 'loso' else 'tstrp'
    oi = list(OBJECTIVES).index(objective)
    return sample_configs(model, objective, mi, oi)[:N_SEARCH_CONFIGS]


def _anchor_cfg(model: str) -> dict:
    """Committed anchor config dict (augr=1.0, cfg_id=0)."""
    c = dict(ANCHOR[model])
    c['augr'] = 1.0
    c['cfg_id'] = 0
    return c


# ---------------------------------------------------------------------------
# scoring (one trained checkpoint -> per-draw accuracies for one objective)
# ---------------------------------------------------------------------------
def _features_from_long(df_long, cache, extract_fixed_features, n_jobs):
    """Extract the cache's top-20 (sanitised) feature matrix from a tsfresh long-format frame."""
    X_raw = extract_fixed_features(df_long, cache['top_20_features_raw'], n_jobs)
    X_san = X_raw.copy()
    X_san.columns = [re.sub(r'[^\w]', '_', c) for c in X_san.columns]
    return X_san[cache['top_20_features_sanitized']].values


def score(model_name, region, objective, init_seed, split_seed, fold, n_folds, cv_mode, part_dropout,
          ld, ed, ped, fb, alpha, ncop, augr, hp_tag, include_subjects, n_draws, eval_val,
          outer, inner, cfg_id=-1, n_jobs=1, dataset_dir='dataset', device=None) -> dict:
    """Score one trained generator for one objective over `n_draws` synthetic resamples.

    Loads the matching checkpoint (built by `core.train` with the same flags), reuses/builds the
    trtr cache, and for each draw generates synthetic signals, trains the stacking classifier, and
    evaluates on the real test fold (TSTR / LOSO) or real+synthetic on test (TSTR+ / LOSO+). When
    `eval_val` is set (kfold inner selection) the classifier is also scored on the held-out
    validation split. The classifier seed is fixed across draws so draw-to-draw variance isolates
    synthetic-sampling noise.

    :return: result dict with per-draw `test_acc`/`test_f1`/`val_acc` lists and the run identity.
    """
    import torch

    from core.data import get_split
    from core.tstr import (build_run_id, df_to_df_long, evaluate_classifier, extract_fixed_features,
                           generate_synthetic_signals, load_cache, load_model,
                           train_stacking_classifier, trtr)

    device = device or torch.device('cpu')
    mode = OBJECTIVES[objective][0]   # 'tstr' or 'tstr_plus'
    phys = _phys_flags(model_name)

    run_id = build_run_id(model_name, region, init_seed, fold, ld, ed, ped, fb, alpha, ncop,
                          cv_mode=cv_mode, part_dropout=part_dropout,
                          phys_residual=phys['phys_residual'], phys_prep=phys['phys_prep'],
                          include_subjects=include_subjects, hp_tag=hp_tag)

    sub_tag = subset_tag(include_subjects)
    cache = load_cache(region, init_seed, split_seed, fold, n_folds, cv_mode=cv_mode, sub_tag=sub_tag)
    if cache is None:
        cache = trtr(dataset_dir, region, n_jobs, init_seed, split_seed, fold, n_folds=n_folds,
                     cv_mode=cv_mode, include_subjects=include_subjects)

    model, ckpt_stats = load_model(model_name, run_id, device)
    participant_idx = model.null_part_idx if (cv_mode == 'loso' and getattr(model, '_cond_part', False)) else None
    n_synth = int(cache['n_train'] * augr) if mode == 'tstr_plus' else cache['n_train']

    # held-out validation features for kfold inner selection (test fold stays untouched)
    val_feats = None
    if eval_val:
        df = load_dataset(dataset_dir)
        df = df[df['region'] == region].reset_index(drop=True)
        _, df_val, _ = get_split(df, cv_mode=cv_mode, fold=fold - 1, n_folds=n_folds,
                                 split_seed=split_seed, include_subjects=include_subjects)
        df_long_val, y_val = df_to_df_long(df_val)
        X_val = _features_from_long(df_long_val, cache, extract_fixed_features, n_jobs)
        val_feats = (X_val, y_val.values)

    test_acc, test_f1, val_acc = [], [], []
    for d in range(n_draws):
        draw_seed = init_seed + 1000 * d   # vary only the draw; offset keeps it off the seed grid
        synth, synth_y = generate_synthetic_signals(model, model_name, n_synth, ckpt_stats, device,
                                                    draw_seed, participant_idx=participant_idx)
        n, _C, T = synth.shape
        df_long = pd.DataFrame({'id': np.repeat(np.arange(n), T), 'time': np.tile(np.arange(T), n),
                                'Humidity': synth[:, 0, :].ravel(), 'Temperature': synth[:, 1, :].ravel()})
        X_synth = _features_from_long(df_long, cache, extract_fixed_features, n_jobs)
        if mode == 'tstr_plus':
            X_tr = np.concatenate([cache['X_train_top'], X_synth])
            y_tr = np.concatenate([cache['y_train'], synth_y])
        else:
            X_tr, y_tr = X_synth, synth_y
        clf = train_stacking_classifier(X_tr, y_tr, init_seed, n_jobs=n_jobs)   # fixed seed -> draw isolates sampling noise
        m = evaluate_classifier(clf, cache['X_test_top'], cache['y_test'])
        test_acc.append(m['accuracy'])
        test_f1.append(m['f1_weighted'])
        if val_feats is not None:
            val_acc.append(evaluate_classifier(clf, val_feats[0], val_feats[1])['accuracy'])

    return {'model': model_name, 'objective': objective, 'mode': mode, 'region': region,
            'cv_mode': cv_mode, 'init_seed': init_seed, 'split_seed': split_seed, 'fold': fold,
            'n_folds': n_folds, 'part_dropout': part_dropout, 'outer': outer, 'inner': inner,
            'cfg_id': cfg_id,
            'ld': ld, 'ed': ed, 'ped': ped, 'fb': fb, 'alpha': alpha, 'ncop': ncop, 'augr': augr,
            'hp_tag': hp_tag, 'include': ','.join(include_subjects), 'n_draws': n_draws,
            'eval_val': bool(eval_val), 'n_train': cache['n_train'], 'n_synthetic': n_synth,
            'test_acc': test_acc, 'test_f1': test_f1, 'val_acc': val_acc,
            'trtr_acc': cache['trtr_metrics']['accuracy'], 'run_id': run_id}


def run_score(args) -> None:
    """CLI: score one checkpoint for one objective and write the result JSON under results/tuning/<stage>/."""
    include = tuple(x.strip() for x in args.include.split(',') if x.strip() and x.strip() != '-')
    result = score(args.model, args.region, args.objective, args.seed, args.split_seed, args.fold,
                   args.nf, OBJECTIVES[args.objective][1], args.pd, args.ld, args.ed, args.ped,
                   args.fb, args.alpha, args.ncop, args.augr, args.hptag, include, args.n_draws,
                   bool(args.eval_val), args.outer, args.inner, cfg_id=args.cfg_id,
                   n_jobs=args.n_jobs, dataset_dir=args.dataset_dir)
    out_dir = f'results/tuning/{args.stage}'
    os.makedirs(out_dir, exist_ok=True)
    out = f"{out_dir}/{args.model}_{result['run_id']}_{args.objective}.json"
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)
    acc = float(np.mean(result['test_acc']))
    print(f"[SCORE] {args.stage} {args.objective} {args.model} {args.region} outer={args.outer} "
          f"inner={args.inner} seed={args.seed} -> test_acc={acc:.4f} (n_draws={args.n_draws}) -> {out}")


# ---------------------------------------------------------------------------
# manifest generation (one row per SLURM array task)
# ---------------------------------------------------------------------------
PILOT_COLS = ['model', 'cv', 'region', 'cfg_id', 'hptag', 'ld', 'ed', 'ped', 'fb', 'bm', 'alpha',
              'ncop', 'augr', 'pd', 'seed', 'fold', 'nf', 'n_draws']
SEARCH_COLS = ['model', 'cv', 'region', 'cfg_id', 'hptag', 'ld', 'ed', 'ped', 'fb', 'bm', 'alpha',
               'ncop', 'augr', 'pd', 'seed', 'outer', 'inner', 'include', 'nf', 'eval_val', 'n_draws']
FINAL_COLS = ['model', 'objective', 'cv', 'region', 'which', 'cfg_id', 'hptag', 'ld', 'ed', 'ped',
              'fb', 'bm', 'alpha', 'ncop', 'augr', 'pd', 'seed', 'fold', 'nf', 'n_draws']


def _write_tsv(cols: list[str], rows: list[list], out: str) -> None:
    """Write a header + tab-separated rows manifest, the format the SLURM array scripts `sed`-read."""
    with open(out, 'w') as f:
        f.write('\t'.join(cols) + '\n')
        for r in rows:
            f.write('\t'.join(str(x) for x in r) + '\n')
    print(f'[MANIFEST] wrote {len(rows)} rows to {out}')


def pilot_manifest(out: str, dataset_dir: str = 'dataset') -> None:
    """Enumerate the pilot grid: models x cv x regions x (anchor+diverse) x seeds x all folds.
    One row trains once and is scored for BOTH objectives of its cv (tstr/tstr+ or loso/loso+)."""
    nloso = n_loso_folds(load_dataset(dataset_dir))
    rows = []
    for mi, model in enumerate(MODELS):
        cfgs = _pilot_configs(model, mi)
        for cv in ('kfold', 'loso'):
            nf = 5 if cv == 'kfold' else nloso
            pd_ = 0.0 if cv == 'kfold' else 0.1   # null-token dropout required for LOSO generation
            for region in REGIONS:
                for cid, c in enumerate(cfgs):
                    hptag = f'p0{MODEL0[model]}c{cid:02d}'
                    for seed in PILOT_SEEDS:
                        for fold in range(1, nf + 1):
                            rows.append([model, cv, region, cid, hptag, c['ld'], c['ed'], c['ped'],
                                         c['fb'], c['bm'], c['alpha'], c['ncop'], c['augr'], pd_,
                                         seed, fold, nf, N_DRAWS_PILOT])
    _write_tsv(PILOT_COLS, rows, out)


def nsearch_manifest(out: str, dataset_dir: str = 'dataset') -> None:
    """Enumerate the nested Stage-A inner-selection grid (outer test fold NEVER trained on).

    LOSO: restrict to the outer-train pool (`include`=all but outer) and sweep every inner subject
    -> full inner-LOSO average. kfold: train on the fold's train split, select on its val split
    (`eval_val`=1); the outer test fold is untouched.
    """
    df = load_dataset(dataset_dir)
    rows = []
    for mi, model in enumerate(MODELS):
        for region in REGIONS:
            subs = sorted(df[df['region'] == region]['participant'].unique())
            n = len(subs)
            # kfold family (tstr / tstr+): select on the per-fold validation split
            for cid, c in enumerate(_search_configs(model, 'kfold', mi)):
                hptag = f'ns{MODEL0[model]}kc{cid:02d}'
                for seed in SELECT_SEEDS:
                    for outer in range(1, 6):
                        rows.append([model, 'kfold', region, cid, hptag, c['ld'], c['ed'], c['ped'],
                                     c['fb'], c['bm'], c['alpha'], c['ncop'], c['augr'], 0.0, seed,
                                     outer, -1, '-', 5, 1, N_DRAWS_SEL])
            # loso family (loso / loso+): full inner-LOSO over the outer-train pool
            for cid, c in enumerate(_search_configs(model, 'loso', mi)):
                hptag = f'ns{MODEL0[model]}lc{cid:02d}'
                for seed in SELECT_SEEDS:
                    for outer in range(1, n + 1):
                        pool = [s for i, s in enumerate(subs) if i != outer - 1]
                        include = ','.join(pool)
                        nfp = len(pool)
                        for inner in range(1, nfp + 1):
                            rows.append([model, 'loso', region, cid, hptag, c['ld'], c['ed'],
                                         c['ped'], c['fb'], c['bm'], c['alpha'], c['ncop'],
                                         c['augr'], 0.1, seed, outer, inner, include, nfp, 0,
                                         N_DRAWS_SEL])
    _write_tsv(SEARCH_COLS, rows, out)


def nfinal_manifest(winners_path: str, out: str, dataset_dir: str = 'dataset') -> None:
    """Expand a winners JSON (per-outer-fold selected config + anchor) into the Stage-B grid:
    each outer fold's selected config and the fixed anchor, refit on the untouched outer test
    fold over the full seed grid."""
    nloso = n_loso_folds(load_dataset(dataset_dir))
    winners = json.load(open(winners_path))
    rows = []
    for key, w in winners.items():
        model, objective, region = key.split('|')
        _mode, cv, pd_ = OBJECTIVES[objective]
        nf = nloso if cv == 'loso' else 5
        for outer_str, cfg in w['per_fold'].items():
            outer = int(outer_str)
            hptag = f"nfs{MODEL0[model]}{objective}c{cfg['cfg_id']:02d}"
            for seed in FULL_SEEDS:
                rows.append([model, objective, cv, region, 'sel', cfg['cfg_id'], hptag, cfg['ld'],
                             cfg['ed'], cfg['ped'], cfg['fb'], cfg['bm'], cfg['alpha'], cfg['ncop'],
                             cfg['augr'], pd_, seed, outer, nf, N_DRAWS_FINAL])
        anc = w['anchor']
        hptag = f'nfa{MODEL0[model]}{objective}'
        for outer in range(1, nf + 1):
            for seed in FULL_SEEDS:
                rows.append([model, objective, cv, region, 'anchor', 0, hptag, anc['ld'], anc['ed'],
                             anc['ped'], anc['fb'], anc['bm'], anc['alpha'], anc['ncop'],
                             anc['augr'], pd_, seed, outer, nf, N_DRAWS_FINAL])
    _write_tsv(FINAL_COLS, rows, out)


# ---------------------------------------------------------------------------
# analysis: pilot report (noise floor + MDD + gate)
# ---------------------------------------------------------------------------
def _load_jsons(directory: str) -> list[dict]:
    """Load every *.json result written by `run_score` in `directory`."""
    out = []
    for p in sorted(glob.glob(os.path.join(directory, '*.json'))):
        with open(p) as f:
            out.append(json.load(f))
    return out


def pilot_report(directory: str = 'results/tuning/pilot', out: str = 'results/tuning/pilot_report.csv') -> pd.DataFrame:
    """Decompose the pilot variance and emit the noise floor, MDD and gate verdict per (model, objective, region).

    For each config: per-(fold,seed) mean over draws -> a[fold,seed]; fold means -> headline; the
    headline SE is the across-fold SE of the anchor. sigma_draw/seed/fold are mean within-group
    SDs. MDD = 1.96 * sqrt(2) * SE_headline (unpaired two-config). The gate compares the config
    spread (max-min headline across the pilot configs) to the MDD.
    """
    rows = _load_jsons(directory)
    recs = []
    by = {}
    for r in rows:
        by.setdefault((r['model'], r['objective'], r['region']), []).append(r)

    for (model, objective, region), rs in sorted(by.items()):
        # headline per config = mean over folds of (mean over seeds of (mean over draws))
        cfg_headline = {}
        draw_sds, seed_sds, fold_sds = [], [], []
        for cid in sorted({r['cfg_id'] for r in rs}):
            crs = [r for r in rs if r['cfg_id'] == cid]
            # a[fold][seed] = mean over draws
            cell = {}
            for r in crs:
                cell.setdefault(r['fold'], {})[r['init_seed']] = float(np.mean(r['test_acc']))
                if len(r['test_acc']) > 1:
                    draw_sds.append(float(np.std(r['test_acc'], ddof=1)))
            fold_means = []
            for fold, seedmap in cell.items():
                vals = list(seedmap.values())
                fold_means.append(np.mean(vals))
                if len(vals) > 1:
                    seed_sds.append(float(np.std(vals, ddof=1)))
            if len(fold_means) > 1:
                fold_sds.append(float(np.std(fold_means, ddof=1)))
            cfg_headline[cid] = (float(np.mean(fold_means)), fold_means)

        anchor_head, anchor_folds = cfg_headline.get(0, (np.nan, []))
        se_headline = float(np.std(anchor_folds, ddof=1) / np.sqrt(len(anchor_folds))) if len(anchor_folds) > 1 else np.nan
        mdd = float(1.96 * np.sqrt(2) * se_headline) if not np.isnan(se_headline) else np.nan
        heads = [h for h, _ in cfg_headline.values()]
        spread = float(max(heads) - min(heads)) if heads else np.nan
        gate = 'config-sensitive' if (not np.isnan(mdd) and spread > mdd) else 'within-noise'
        recs.append({'model': model, 'objective': objective, 'region': region,
                     'anchor_headline': round(anchor_head, 4), 'se_headline': round(se_headline, 4),
                     'mdd': round(mdd, 4), 'config_spread': round(spread, 4),
                     'sigma_draw': round(float(np.mean(draw_sds)), 4) if draw_sds else np.nan,
                     'sigma_seed': round(float(np.mean(seed_sds)), 4) if seed_sds else np.nan,
                     'sigma_fold': round(float(np.mean(fold_sds)), 4) if fold_sds else np.nan,
                     'n_configs': len(cfg_headline), 'gate': gate})
    df = pd.DataFrame(recs)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f'[PILOT] {len(df)} (model,objective,region) cells -> {out}')
    if not df.empty:
        print(df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# analysis: nested selection (1-SE rule, per outer fold)
# ---------------------------------------------------------------------------
def _select_units(r: dict) -> list[float]:
    """Per-draw selection accuracies for a Stage-A row: val_acc for kfold (eval_val), else test_acc."""
    return r['val_acc'] if r['eval_val'] and r['val_acc'] else r['test_acc']


def select_nested(directory: str = 'results/tuning/nsearch',
                  out: str = 'results/tuning/nested_winners.json') -> dict:
    """Pick, per (model, objective, region, outer fold), the config chosen by the 1-SE rule.

    Inner score per config = mean over its selection units (kfold: val draws x seeds; loso: test
    draws x inner subjects x seeds). Keep all configs within one standard error of the best, then
    take the simplest (`_complexity`). Writes a winners JSON consumed by `nfinal_manifest`.
    """
    rows = _load_jsons(directory)
    # objective is a property of (cv, mode); a Stage-A row covers both objectives of its cv.
    winners: dict = {}
    for model in MODELS:
        for region in REGIONS:
            for objective, (mode, cv, _pd) in OBJECTIVES.items():
                rs = [r for r in rows if r['model'] == model and r['region'] == region and r['cv_mode'] == cv]
                # a Stage-A row's `mode` reflects how it was scored; pick rows matching this objective's mode
                rs = [r for r in rs if r['mode'] == mode]
                if not rs:
                    continue
                # authoritative config table (incl. beta_max, which score() does not record in the json)
                cfg_table = {i: dict(c, cfg_id=i) for i, c in enumerate(_search_configs(model, cv, MODELS.index(model)))}
                per_fold = {}
                for outer in sorted({r['outer'] for r in rs}):
                    ors = [r for r in rs if r['outer'] == outer]
                    cfg_units, cfg_meta = {}, {}
                    for cid in sorted({r['cfg_id'] for r in ors}):
                        units = []
                        for r in [x for x in ors if x['cfg_id'] == cid]:
                            units.extend(_select_units(r))
                        if units:
                            cfg_units[cid] = np.asarray(units, dtype=float)
                            cfg_meta[cid] = cfg_table[cid]
                    if not cfg_units:
                        continue
                    means = {cid: float(u.mean()) for cid, u in cfg_units.items()}
                    best = max(means, key=means.get)
                    bu = cfg_units[best]
                    se = float(bu.std(ddof=1) / np.sqrt(len(bu))) if len(bu) > 1 else 0.0
                    thresh = means[best] - se
                    keep = [cid for cid in cfg_units if means[cid] >= thresh]
                    chosen = min(keep, key=lambda cid: _complexity(cfg_meta[cid]))
                    per_fold[str(outer)] = cfg_meta[chosen] | {'inner_score': round(means[chosen], 4),
                                                               'best_score': round(means[best], 4)}
                if per_fold:
                    anc = _anchor_cfg(model)
                    if mode == 'tstr_plus':  # keep the anchor's committed augmentation ratio
                        anc['augr'] = 1.0
                    winners[f'{model}|{objective}|{region}'] = {'per_fold': per_fold, 'anchor': anc}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(winners, f, indent=2)
    print(f'[SELECT] wrote {len(winners)} (model,objective,region) winners -> {out}')
    for k, w in winners.items():
        picks = {fold: c['cfg_id'] for fold, c in w['per_fold'].items()}
        print(f'  {k}: per-fold cfg_ids={picks}')
    return winners


# ---------------------------------------------------------------------------
# analysis: final report (headline +/- CI, paired test vs anchor, MDD gate)
# ---------------------------------------------------------------------------
def _paired_test(deltas: np.ndarray) -> float:
    """Two-sided p-value that the paired (sel - anchor) differences are non-zero (Wilcoxon, else t)."""
    deltas = np.asarray(deltas, dtype=float)
    if len(deltas) < 2 or np.allclose(deltas, 0):
        return float('nan')
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(deltas).pvalue)
    except Exception:
        from statistics import NormalDist
        m, s = float(np.mean(deltas)), float(np.std(deltas, ddof=1))
        if s == 0:
            return float('nan')
        z = m / (s / np.sqrt(len(deltas)))
        return float(2 * (1 - NormalDist().cdf(abs(z))))


def final_report(directory: str = 'results/tuning/nfinal',
                 pilot_csv: str = 'results/tuning/pilot_report.csv',
                 out: str = 'results/tuning/nested_final_report.csv') -> pd.DataFrame:
    """Headline +/- CI for selected vs anchor per (model, objective, region), with a paired test.

    Pairs the inner-selected config against the anchor by (outer fold, seed) on the untouched outer
    test fold (draws averaged), tests the differences, and flags `improved` only when the gain is
    positive, significant (p<0.05) and clears the pilot MDD.
    """
    rows = _load_jsons(directory)
    mdd_lookup = {}
    if os.path.exists(pilot_csv):
        pdf = pd.read_csv(pilot_csv)
        for _, r in pdf.iterrows():
            mdd_lookup[(r['model'], r['objective'], r['region'])] = float(r['mdd'])

    # group by (model,objective,region) -> {sel|anchor: {(outer,seed): mean test acc over draws}}
    grp = {}
    for r in rows:
        key = (r['model'], r['objective'], r['region'])
        which = 'anchor' if r['hp_tag'].startswith('nfa') else 'sel'
        grp.setdefault(key, {'sel': {}, 'anchor': {}})
        grp[key][which][(r['outer'], r['init_seed'])] = float(np.mean(r['test_acc']))

    recs = []
    for key, d in sorted(grp.items()):
        sel, anc = d['sel'], d['anchor']
        common = sorted(set(sel) & set(anc))
        sel_all = list(sel.values())
        anc_all = list(anc.values())
        deltas = np.array([sel[k] - anc[k] for k in common]) if common else np.array([])
        sel_mean = float(np.mean(sel_all)) if sel_all else np.nan
        anc_mean = float(np.mean(anc_all)) if anc_all else np.nan
        delta = float(np.mean(deltas)) if len(deltas) else np.nan
        ci = float(1.96 * np.std(deltas, ddof=1) / np.sqrt(len(deltas))) if len(deltas) > 1 else np.nan
        p = _paired_test(deltas)
        mdd = mdd_lookup.get(key, np.nan)
        improved = bool(delta > 0 and (not np.isnan(p) and p < 0.05) and (np.isnan(mdd) or delta > mdd))
        recs.append({'model': key[0], 'objective': key[1], 'region': key[2],
                     'anchor_acc': round(anc_mean, 4), 'selected_acc': round(sel_mean, 4),
                     'delta': round(delta, 4), 'delta_ci95': round(ci, 4) if not np.isnan(ci) else np.nan,
                     'p_value': round(p, 4) if not np.isnan(p) else np.nan,
                     'mdd': round(mdd, 4) if not np.isnan(mdd) else np.nan,
                     'n_pairs': len(deltas), 'improved': improved})
    df = pd.DataFrame(recs)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f'[FINAL] {len(df)} (model,objective,region) cells -> {out}')
    if not df.empty:
        print(df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the pilot/nested-search/final subcommands."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    s = sub.add_parser('score', help='score one checkpoint for one objective (called per SLURM task)')
    s.add_argument('--stage', required=True, choices=['pilot', 'nsearch', 'nfinal'])
    s.add_argument('--model', required=True, choices=MODELS)
    s.add_argument('--region', required=True, choices=REGIONS)
    s.add_argument('--objective', required=True, choices=list(OBJECTIVES))
    s.add_argument('--seed', type=int, required=True)
    s.add_argument('--split_seed', type=int, default=42)
    s.add_argument('--fold', type=int, required=True)
    s.add_argument('--nf', type=int, required=True)
    s.add_argument('--pd', type=float, default=0.0)
    s.add_argument('--ld', type=int, required=True)
    s.add_argument('--ed', type=int, required=True)
    s.add_argument('--ped', type=int, required=True)
    s.add_argument('--fb', type=float, required=True)
    s.add_argument('--alpha', type=float, required=True)
    s.add_argument('--ncop', type=int, required=True)
    s.add_argument('--augr', type=float, default=1.0)
    s.add_argument('--hptag', required=True)
    s.add_argument('--cfg_id', type=int, default=-1)
    s.add_argument('--include', default='')
    s.add_argument('--outer', type=int, default=-1)
    s.add_argument('--inner', type=int, default=-1)
    s.add_argument('--n_draws', type=int, default=1)
    s.add_argument('--eval_val', type=int, default=0)
    s.add_argument('--n_jobs', type=int, default=1)
    s.add_argument('--dataset_dir', default='dataset')

    m = sub.add_parser('pilot-manifest')
    m.add_argument('--out', required=True)
    se = sub.add_parser('nsearch-manifest')
    se.add_argument('--out', required=True)
    fi = sub.add_parser('nfinal-manifest')
    fi.add_argument('--winners', required=True)
    fi.add_argument('--out', required=True)

    pr = sub.add_parser('pilot-report')
    pr.add_argument('--dir', default='results/tuning/pilot')
    pr.add_argument('--out', default='results/tuning/pilot_report.csv')
    ns = sub.add_parser('nselect')
    ns.add_argument('--dir', default='results/tuning/nsearch')
    ns.add_argument('--out', default='results/tuning/nested_winners.json')
    nr = sub.add_parser('nreport')
    nr.add_argument('--dir', default='results/tuning/nfinal')
    nr.add_argument('--pilot', default='results/tuning/pilot_report.csv')
    nr.add_argument('--out', default='results/tuning/nested_final_report.csv')
    return p.parse_args()


def main() -> None:
    """Dispatch the requested subcommand."""
    args = parse_args()
    if args.cmd == 'score':
        run_score(args)
    elif args.cmd == 'pilot-manifest':
        pilot_manifest(args.out)
    elif args.cmd == 'nsearch-manifest':
        nsearch_manifest(args.out)
    elif args.cmd == 'nfinal-manifest':
        nfinal_manifest(args.winners, args.out)
    elif args.cmd == 'pilot-report':
        pilot_report(args.dir, args.out)
    elif args.cmd == 'nselect':
        select_nested(args.dir, args.out)
    elif args.cmd == 'nreport':
        final_report(args.dir, args.pilot, args.out)


if __name__ == '__main__':
    main()
