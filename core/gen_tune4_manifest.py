"""Phase-4d: generate a TSV manifest for tuning cvae_part + tpinn-res across 4 objectives
(tstr / loso / tstr+ / loso+). Each manifest ROW = one SLURM array task = one
(model, objective, region, hp-config, init_seed, fold). experiments/35_tune4obj.sh reads a row
and runs core.train + core.tstr with those args. Every run is namespaced by --hp_tag
"t4<obj><model0>c<cfg>" so tstr vs tstr+ (same cv_mode) never collide on {run_id}_tstr.json,
and nothing touches the committed result namespace.

Modes:
  search   -> 16 hp-configs/model/objective on the cheap proxy (seeds 0,42 x folds 1,3), 200 ep.
  validate -> best config per (model,objective,region) [read from a winners JSON] at the FULL
              protocol (seeds 0,1,7,42,123 x all folds), 500 ep.

HP search spaces (config 0 = committed anchor, always included):
  cvae_part: latent_dim, embed_dim, part_embed_dim, free_bits, beta_max, alpha, n_copies
  tpinn-res: latent_dim, free_bits, beta_max, alpha, n_copies (embed/ped fixed 8; always --phys_residual
             --phys_prep stdscale; lambda_phys moot)
  + objectives also tune augmentation_ratio (ignored by non-plus objectives).
"""
import argparse
import json
import random

from core.data import load_dataset, n_loso_folds

# objective -> (mode, cv_mode, part_dropout)
OBJECTIVES = {
    'tstr':  ('tstr',      'kfold', 0.0),
    'loso':  ('tstr',      'loso',  0.1),
    'tstrp': ('tstr_plus', 'kfold', 0.0),
    'losop': ('tstr_plus', 'loso',  0.1),
}
MODELS = ['cvae_part', 'tpinn']
MODEL0 = {'cvae_part': 'c', 'tpinn': 't'}
REGIONS = ['mouth', 'nose']
SEARCH_SEEDS = [0, 42]
SEARCH_FOLDS = [1, 3]
FULL_SEEDS = [0, 1, 7, 42, 123]
COLS = ['model', 'objective', 'mode', 'cv', 'pd', 'region', 'hptag', 'ld', 'ed', 'ped',
        'fb', 'bm', 'alpha', 'ncop', 'augr', 'seed', 'fold', 'nf']

# committed anchors (config 0)
ANCHOR = {
    'cvae_part': dict(ld=16, ed=8, ped=8, fb=0.0, bm=0.01, alpha=0.05, ncop=10),
    'tpinn':     dict(ld=16, ed=8, ped=8, fb=0.0, bm=0.01, alpha=0.05, ncop=10),
}
SPACE = {
    'cvae_part': dict(ld=[8, 16, 32], ed=[4, 8, 16], ped=[4, 8, 16], fb=[0.0, 0.1, 0.5],
                      bm=[0.003, 0.01, 0.03, 0.1], alpha=[0.0, 0.05, 0.1], ncop=[1, 5, 10, 20]),
    'tpinn':     dict(ld=[8, 16, 32], ed=[8], ped=[8], fb=[0.0, 0.5],
                      bm=[0.003, 0.01, 0.03], alpha=[0.0, 0.05, 0.1], ncop=[1, 5, 10, 20]),
}
AUGR = [0.5, 1.0, 2.0, 3.0]
N_CONFIGS = 16


def sample_configs(model, objective, mi, oi):
    """Deterministic per (model,objective): config 0 = anchor, rest = seeded random samples."""
    rng = random.Random(1000 * mi + oi)
    sp = SPACE[model]
    plus = objective in ('tstrp', 'losop')
    cfgs = []
    anchor = dict(ANCHOR[model]); anchor['augr'] = 1.0
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
        seen.add(key); cfgs.append(c)
    return cfgs


def row(model, objective, region, cfg, cfg_id, seed, fold, nf):
    mode, cv, pd = OBJECTIVES[objective]
    hptag = f"t4{objective}{MODEL0[model]}c{cfg_id:02d}"
    return [model, objective, mode, cv, pd, region, hptag, cfg['ld'], cfg['ed'], cfg['ped'],
            cfg['fb'], cfg['bm'], cfg['alpha'], cfg['ncop'], cfg['augr'], seed, fold, nf]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['search', 'validate'])
    ap.add_argument('--winners', default='', help='validate: JSON {model|objective|region: cfg dict} of best configs')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    nloso = n_loso_folds(load_dataset('dataset'))
    rows = []
    if args.mode == 'search':
        for mi, model in enumerate(MODELS):
            for oi, objective in enumerate(OBJECTIVES):
                _, cv, _ = OBJECTIVES[objective]
                nf = nloso if cv == 'loso' else 5
                cfgs = sample_configs(model, objective, mi, oi)
                for region in REGIONS:
                    for cfg_id, cfg in enumerate(cfgs):
                        for seed in SEARCH_SEEDS:
                            for fold in SEARCH_FOLDS:
                                rows.append(row(model, objective, region, cfg, cfg_id, seed, fold, nf))
    else:
        winners = json.load(open(args.winners))
        for key, cfgs in winners.items():
            model, objective, region = key.split('|')
            _, cv, _ = OBJECTIVES[objective]
            nf = nloso if cv == 'loso' else 5
            folds = list(range(1, nf + 1))
            cfg_list = cfgs if isinstance(cfgs, list) else [cfgs]   # value may be a single cfg or a list (winner + anchor)
            seen_ids = set()
            for cfg in cfg_list:
                cfg_id = cfg['cfg_id']
                if cfg_id in seen_ids:
                    continue   # dedup: winner may equal anchor
                seen_ids.add(cfg_id)
                for seed in FULL_SEEDS:
                    for fold in folds:
                        rows.append(row(model, objective, region, cfg, cfg_id, seed, fold, nf))

    with open(args.out, 'w') as f:
        f.write('\t'.join(COLS) + '\n')
        for r in rows:
            f.write('\t'.join(str(x) for x in r) + '\n')
    print(f'[MANIFEST] {args.mode}: wrote {len(rows)} rows to {args.out} (nloso={nloso})')


if __name__ == '__main__':
    main()
