import argparse
import json
import os
import random
import re

import numpy as np
import pandas as pd

from core.data import load_dataset, n_loso_folds

## Optuna tuning
def _empty_device_cache(device) -> None:
    """Free cached cuda/mps memory for `device`."""
    import torch
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()

def _train_model(model_name: str, params: dict, region: str, dataset_dir: str, epochs: int, seed: int, fold: int, split_seed: int, n_folds: int, device, cv_mode: str = 'kfold', part_dropout: float = 0.0):
    """Train a (c)VAE generator on the same fold split the trtr cache scores against.

    :param model_name: one of 'vae', 'cvae', 'cvae_part'.
    :param params: hyperparameter dict (latent_dim, lr, free_bits, beta_max, alpha, ...).
    :param fold: 1-based fold; get_split is called with fold-1 to match core.tstr (no leakage).
    :param cv_mode: 'kfold' or 'loso' (loso holds out subject `fold`).
    :param part_dropout: probability of the null participant token (cvae_part unseen-subject gen).
    :return: (trained model, training-set normalization stats).
    """
    import torch
    from torch.utils.data import DataLoader
    from core.data import BreathDataset, get_split
    from core.train import evaluate, train_vae_one_epoch
    from core.utils import make_generator, seed_everything, seed_worker
    from models.vae import CVAE, VAE

    seed_everything(seed)
    g = make_generator(seed)

    df = load_dataset(dataset_dir)
    df = df[df['region'] == region].reset_index(drop=True)
    # SAME split as the trtr cache (core.tstr uses fold-1) -> no train/test leakage. cv_mode=loso
    # holds out subject `fold`; part_dropout>0 allocates the null token for unseen-subject generation.
    df_train, df_val, _ = get_split(df, cv_mode=cv_mode, fold=fold - 1, n_folds=n_folds, split_seed=split_seed)
    alpha = params.get('alpha', 0.0)
    n_copies = params.get('n_copies', 1) if alpha > 0 else 1
    train_ds = BreathDataset(df_train, alpha=alpha, n_copies=n_copies)
    val_ds = BreathDataset(df_val, stats=train_ds.stats)
    train_loader = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True, drop_last=False, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_ds, batch_size=params['batch_size'], shuffle=False)

    if model_name == 'cvae_part':
        num_participants = df['participant'].nunique()   # embedding must cover all subjects (was defaulting to 3 -> IndexError)
        model = CVAE(latent_dim=params['latent_dim'], embed_dim=params['embed_dim'],
                     condition_on_participant=True, num_participants=num_participants,
                     part_embed_dim=params['part_embed_dim'], part_dropout=part_dropout).to(device)
    elif model_name == 'cvae':
        model = CVAE(latent_dim=params['latent_dim'], embed_dim=params['embed_dim']).to(device)
    else:
        model = VAE(latent_dim=params['latent_dim']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    warmup = max(1, int(params['warmup_frac'] * epochs))
    conditional = model_name in ('cvae', 'cvae_part')
    use_participant = model_name == 'cvae_part'
    for epoch in range(1, epochs + 1):
        train_vae_one_epoch(model, train_loader, optimizer, epoch, warmup, device, params['free_bits'], conditional=conditional, use_participant=use_participant, beta_max=params['beta_max'])
        evaluate(model, val_loader, epoch, warmup, device, params['free_bits'], conditional=conditional, use_participant=use_participant, beta_max=params['beta_max'])
        scheduler.step()

    return model, train_ds.stats

def _tstr_accuracy(model, model_name: str, region: str, dataset_dir: str, n_jobs: int,
                   seed: int, fold: int, split_seed: int, n_folds: int, n_synthetic, device,
                   cv_mode: str = 'kfold') -> float:
    """Score a trained generator's TSTR accuracy against the matched real-test cache.

    Generates synthetic signals, extracts the cache's top-20 features, trains a stacking
    classifier on them, and evaluates on the cached real test set. Builds the trtr cache
    if it does not already exist.

    :param model: trained generator from `_train_model`.
    :param n_synthetic: number of signals to generate (defaults to the cache's n_train).
    :param cv_mode: under 'loso' the unseen test subject is generated from the null token.
    :return: train-on-synthetic test-on-real accuracy.
    """
    from core.tstr import evaluate_classifier, extract_fixed_features, generate_synthetic_signals, load_cache, train_stacking_classifier, trtr

    cache = load_cache(region, seed, split_seed, fold, n_folds, cv_mode=cv_mode)
    if cache is None:  # standard caches usually exist (results/trtr/*.pkl); build if missing
        cache = trtr(dataset_dir, region, n_jobs, seed, split_seed, fold, n_folds=n_folds, cv_mode=cv_mode)

    n_synth = n_synthetic if n_synthetic is not None else cache['n_train']
    # under LOSO the test subject is unseen -> generate from the null token (cvae_part only)
    participant_idx = model.null_part_idx if (cv_mode == 'loso' and getattr(model, '_cond_part', False)) else None
    synth_signals, synth_labels = generate_synthetic_signals(model, model_name, n_synth, cache['stats'], device, seed, participant_idx=participant_idx)

    n, _C, T = synth_signals.shape
    ids = np.repeat(np.arange(n), T)
    times = np.tile(np.arange(T), n)
    df_long = pd.DataFrame({'id': ids, 'time': times,
                            'Humidity': synth_signals[:, 0, :].ravel(),
                            'Temperature': synth_signals[:, 1, :].ravel()})

    X_synth_raw = extract_fixed_features(df_long, cache['top_20_features_raw'], n_jobs)
    X_san = X_synth_raw.copy()
    X_san.columns = [re.sub(r'[^\w]', '_', col) for col in X_san.columns]
    X_synth_top = X_san[cache['top_20_features_sanitized']].values

    clf = train_stacking_classifier(X_synth_top, synth_labels, seed)
    metrics = evaluate_classifier(clf, cache['X_test_top'], cache['y_test'])
    return float(metrics['accuracy'])

def make_objective(model_name: str, region: str, dataset_dir: str = 'dataset', epochs: int = 200,
                   seeds: tuple = (0, 42), folds: tuple = (1, 3), split_seed: int = 42,
                   n_folds: int = 5, n_synthetic=None, n_jobs: int = 4, device=None,
                   cv_mode: str = 'kfold', part_dropout: float = 0.0):
    """Build an Optuna objective closure that maximizes mean TSTR accuracy.

    :param model_name: one of 'vae', 'cvae', 'cvae_part'.
    :param seeds: training seeds and `folds` the fold ids; their product is the eval grid.
    :param n_synthetic: number of synthetic signals per eval (None -> cache n_train).
    :return: objective(trial) -> float for `study.optimize`.
    """
    import gc

    import optuna
    import torch

    device = device or torch.device('cpu')
    evals = [(f, s) for f in folds for s in seeds]   # (fold, seed) grid, matched train/test splits

    def objective(trial) -> float:
        """Sample hyperparameters, train + score over the eval grid, report intermediate means.

        :param trial: Optuna trial supplying the hyperparameter suggestions.
        :return: mean TSTR accuracy across the (fold, seed) grid (to maximize).
        """
        params = {'latent_dim': trial.suggest_categorical('latent_dim', [8, 16, 32, 64]),
                  'free_bits': trial.suggest_float('free_bits', 0.0, 1.0),
                  'beta_max': trial.suggest_float('beta_max', 1e-3, 1e-1, log=True),
                  'lr': trial.suggest_float('lr', 1e-4, 3e-3, log=True),
                  'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
                  'warmup_frac': trial.suggest_float('warmup_frac', 0.2, 0.6),
                  # jitter augmentation - the grid's biggest TSTR lever (alpha=0.05, n_copies=10 was best)
                  'alpha': trial.suggest_float('alpha', 0.0, 0.15),
                  'n_copies': trial.suggest_categorical('n_copies', [1, 5, 10, 20])}
        if model_name in ('cvae', 'cvae_part'):
            params['embed_dim'] = trial.suggest_categorical('embed_dim', [4, 8, 16, 32])
        if model_name == 'cvae_part':
            params['part_embed_dim'] = trial.suggest_categorical('part_embed_dim', [4, 8, 16, 32])

        accs = []
        model = None
        try:
            for i, (fold, seed) in enumerate(evals):
                try:
                    model, _stats = _train_model(model_name, params, region, dataset_dir, epochs, seed, fold, split_seed, n_folds, device, cv_mode, part_dropout)
                    acc = _tstr_accuracy(model, model_name, region, dataset_dir, n_jobs, seed, fold, split_seed, n_folds, n_synthetic, device, cv_mode)
                except (RuntimeError, ValueError) as e:
                    raise optuna.TrialPruned() from e
                finally:
                    if model is not None:
                        model.zero_grad(set_to_none=True)
                        del model
                        model = None
                    gc.collect()
                    _empty_device_cache(device)

                accs.append(acc)
                trial.report(float(np.mean(accs)), step=i)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return float(np.mean(accs))
        finally:
            gc.collect()
            _empty_device_cache(device)

    return objective

def _log_callback(study, trial) -> None:
    """Print the trial outcome and best-so-far accuracy after each trial."""
    import optuna
    try:
        best = f'{study.best_value:.4f}'
        best_params = study.best_params
    except ValueError:
        best, best_params = 'n/a', {}
    if trial.state == optuna.trial.TrialState.PRUNED:
        print(f'[TUNE] trial {trial.number} pruned | best so far: {best}', flush=True)
    elif trial.value is None:
        print(f'[TUNE] trial {trial.number} failed | best so far: {best}', flush=True)
    else:
        print(f'[TUNE] trial {trial.number} done | acc={trial.value:.4f} | params={trial.params} | best so far: {best} | best_params: {best_params}', flush=True)

def run_search(model_name: str, region: str, n_trials: int = 60, epochs: int = 200, seeds: tuple = (0, 42),
               folds: tuple = (1, 3), split_seed: int = 42, n_folds: int = 5, n_synthetic=None,
               dataset_dir: str = 'dataset', n_jobs: int = 4, sampler_seed: int = 42, device=None,
               cv_mode: str = 'kfold', part_dropout: float = 0.0):
    """Run the TPE TSTR search and persist the trials CSV and best-params JSON.

    Uses a JournalFileBackend storage so a SLURM array can share one study; loso gets
    its own study suffix to keep kfold paths back-compatible.

    :param n_trials: number of Optuna trials to run.
    :param sampler_seed: seed for the TPE sampler.
    :return: the completed Optuna study.
    """
    import optuna

    os.makedirs('results/tuning', exist_ok=True)
    sfx = '' if cv_mode == 'kfold' else f'_{cv_mode}'   # keep kfold paths back-compatible; loso gets its own study
    # JournalFileBackend supports concurrent workers -> a SLURM array can share one study
    storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(f'results/tuning/{model_name}_{region}{sfx}_v4.log'))
    sampler = optuna.samplers.TPESampler(seed=sampler_seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1)
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner,
                                study_name=f'{model_name}_tstr_{region}{sfx}',
                                storage=storage, load_if_exists=True)
    objective = make_objective(model_name, region, dataset_dir, epochs, seeds, folds, split_seed, n_folds, n_synthetic, n_jobs, device, cv_mode, part_dropout)
    study.optimize(objective, n_trials=n_trials, callbacks=[_log_callback], show_progress_bar=False)

    trials_path = f'results/tuning/{model_name}_{region}{sfx}_trials.csv'
    best_path = f'results/tuning/{model_name}_{region}{sfx}_best.json'
    study.trials_dataframe().to_csv(trials_path, index=False)
    with open(best_path, 'w') as f:
        json.dump({'model': model_name, 'region': region, 'cv_mode': cv_mode, 'part_dropout': part_dropout,
                   'seeds': list(seeds), 'folds': list(folds),
                   'search_epochs': epochs, 'best_accuracy': study.best_value, 'best_params': study.best_params}, f, indent=2)
    print(f'[TUNE] Best search-TSTR accuracy: {study.best_value:.4f}')
    print(f'[TUNE] Best params: {study.best_params}')
    print(f'[TUNE] Trials saved to {trials_path}\n[TUNE] Best params saved to {best_path}')
    return study

def run_optuna(args) -> None:
    """Set up the CPU/torch threads and launch `run_search` from parsed `optuna` args."""
    import torch

    device = torch.device('cpu')
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    print(f'[TUNE] Model: {args.model} | Region: {args.region} | Trials: {args.n_trials} | SearchEpochs: {args.epochs} '
          f'| Folds: {args.folds} | Seeds: {args.seeds} | Device: {device} | torch_threads: {torch.get_num_threads()}')
    run_search(model_name=args.model, region=args.region, n_trials=args.n_trials, epochs=args.epochs,
               seeds=tuple(args.seeds), folds=tuple(args.folds), split_seed=args.split_seed, n_folds=args.n_folds,
               n_synthetic=args.n_synthetic, dataset_dir=args.dataset_dir, n_jobs=args.n_jobs,
               sampler_seed=args.sampler_seed, device=device, cv_mode=args.cv_mode, part_dropout=args.part_dropout)


## Phase-4d 4-objective sweep manifest generation
# objective -> (mode, cv_mode, part_dropout)
OBJECTIVES = {
    'tstr': ('tstr',      'kfold', 0.0),
    'loso': ('tstr',      'loso',  0.1),
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
    """Deterministic per (model,objective): config 0 = anchor, rest = seeded random samples."""
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


def row(model, objective, region, cfg, cfg_id, seed, fold, nf):
    """Build one manifest row (a list ordered like COLS) for the given config and run id."""
    mode, cv, pd = OBJECTIVES[objective]
    hptag = f"t4{objective}{MODEL0[model]}c{cfg_id:02d}"
    return [model, objective, mode, cv, pd, region, hptag, cfg['ld'], cfg['ed'], cfg['ped'],
            cfg['fb'], cfg['bm'], cfg['alpha'], cfg['ncop'], cfg['augr'], seed, fold, nf]


def run_manifest(args) -> None:
    """Write the TSV tuning sweep manifest (one row per SLURM array task) to `args.out`.

    'search' mode enumerates models x objectives x regions x configs x seeds x folds on
    the cheap proxy; 'validate' mode reads winning configs from `args.winners` and expands
    them to the full seed/fold protocol.

    :param args: parsed `manifest` args (mode, winners, out).
    """
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    o = sub.add_parser('optuna')
    o.add_argument('--model', required=True, choices=['vae', 'cvae', 'cvae_part'])
    o.add_argument('--region', required=True, choices=['mouth', 'nose'])
    o.add_argument('--n_trials', type=int, default=60)
    o.add_argument('--epochs', type=int, default=200)
    o.add_argument('--seeds', type=int, nargs='+', default=[0, 42])
    o.add_argument('--folds', type=int, nargs='+', default=[1, 3])
    o.add_argument('--split_seed', type=int, default=42)
    o.add_argument('--n_folds', type=int, default=5)
    o.add_argument('--cv_mode', choices=['kfold', 'loso'], default='kfold')
    o.add_argument('--part_dropout', type=float, default=0.0)
    o.add_argument('--n_synthetic', type=int, default=None)
    o.add_argument('--n_jobs', type=int, default=4)
    o.add_argument('--dataset_dir', default='dataset')
    o.add_argument('--sampler_seed', type=int, default=42)
    o.add_argument('--torch_threads', type=int, default=0)

    m = sub.add_parser('manifest')
    m.add_argument('--mode', required=True, choices=['search', 'validate'])
    m.add_argument('--winners', default='')
    m.add_argument('--out', required=True)

    return p.parse_args()

def main() -> None:
    """Dispatch to the optuna search or manifest generation subcommand."""
    args = parse_args()
    if args.cmd == 'optuna':
        run_optuna(args)
    else:
        run_manifest(args)

if __name__ == '__main__':
    main()
