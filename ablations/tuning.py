import argparse
import gc
import json
import os
import re

import numpy as np
import optuna
import pandas as pd
import torch
from torch.utils.data import DataLoader

from core.data import BreathDataset, load_dataset, get_split
from core.train import evaluate, train_vae_one_epoch
from core.tstr import evaluate_classifier, extract_fixed_features, generate_synthetic_signals, load_cache, train_stacking_classifier, trtr
from core.utils import make_generator, seed_everything, seed_worker
from models.vae import CVAE, VAE

# Optuna TSTR tuning for the (c)VAE generators.
# IMPORTANT (leakage): the model is trained on the SAME k-fold split that the trtr cache scores
# against (get_split with fold-1, matching core.tstr.trtr), so synthetic-train / real-test never
# overlap. We tune over a SUBSET of (fold, seed) whose trtr caches already exist (results/trtr/*.pkl),
# then validate the winner at the full 5-fold x 5-seed protocol via experiments/18_tune_validate.sh.

def _empty_device_cache(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()

def _train_model(model_name: str, params: dict, region: str, dataset_dir: str, epochs: int, seed: int, fold: int, split_seed: int, n_folds: int, device: torch.device, cv_mode: str = 'kfold', part_dropout: float = 0.0) -> tuple[torch.nn.Module, dict]:
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

def _tstr_accuracy(model: torch.nn.Module, model_name: str, region: str, dataset_dir: str, n_jobs: int,
                   seed: int, fold: int, split_seed: int, n_folds: int, n_synthetic: int | None, device: torch.device,
                   cv_mode: str = 'kfold') -> float:
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
                   seeds: tuple[int, ...] = (0, 42), folds: tuple[int, ...] = (1, 3), split_seed: int = 42,
                   n_folds: int = 5, n_synthetic: int | None = None, n_jobs: int = 4, device: torch.device | None = None,
                   cv_mode: str = 'kfold', part_dropout: float = 0.0):
    device = device or torch.device('cpu')
    evals = [(f, s) for f in folds for s in seeds]   # (fold, seed) grid, matched train/test splits

    def objective(trial: optuna.Trial) -> float:
        params = {'latent_dim': trial.suggest_categorical('latent_dim', [8, 16, 32, 64]),
                  'free_bits': trial.suggest_float('free_bits', 0.0, 1.0),
                  'beta_max': trial.suggest_float('beta_max', 1e-3, 1e-1, log=True),
                  'lr': trial.suggest_float('lr', 1e-4, 3e-3, log=True),
                  'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
                  'warmup_frac': trial.suggest_float('warmup_frac', 0.2, 0.6),
                  # jitter augmentation — the grid's biggest TSTR lever (alpha=0.05, n_copies=10 was best)
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

def _log_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
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

def run_search(model_name: str, region: str, n_trials: int = 60, epochs: int = 200, seeds: tuple[int, ...] = (0, 42),
               folds: tuple[int, ...] = (1, 3), split_seed: int = 42, n_folds: int = 5, n_synthetic: int | None = None,
               dataset_dir: str = 'dataset', n_jobs: int = 4, sampler_seed: int = 42, device: torch.device | None = None,
               cv_mode: str = 'kfold', part_dropout: float = 0.0) -> optuna.Study:
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

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, choices=['vae', 'cvae', 'cvae_part'])
    p.add_argument('--region', required=True, choices=['mouth', 'nose'])
    p.add_argument('--n_trials', type=int, default=60)
    p.add_argument('--epochs', type=int, default=200, help='reduced epochs for the SEARCH phase (validate winner at 500)')
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 42], help='search subset of init seeds (caches reused)')
    p.add_argument('--folds', type=int, nargs='+', default=[1, 3], help='search subset of k-folds (caches reused)')
    p.add_argument('--split_seed', type=int, default=42)
    p.add_argument('--n_folds', type=int, default=5)
    p.add_argument('--cv_mode', choices=['kfold', 'loso'], default='kfold', help='kfold = in-distribution TSTR; loso = cross-subject TSTR (use --part_dropout 0.1 + --folds over dev subjects)')
    p.add_argument('--part_dropout', type=float, default=0.0, help='cvae_part null-token dropout; set 0.1 for loso so unseen-subject generation works')
    p.add_argument('--n_synthetic', type=int, default=None)
    p.add_argument('--n_jobs', type=int, default=4)
    p.add_argument('--dataset_dir', default='dataset')
    p.add_argument('--sampler_seed', type=int, default=42)
    p.add_argument('--torch_threads', type=int, default=0, help='cap PyTorch CPU threads (0 = leave default)')
    return p.parse_args()

def main() -> None:
    args = parse_args()
    device = torch.device('cpu')
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    print(f'[TUNE] Model: {args.model} | Region: {args.region} | Trials: {args.n_trials} | SearchEpochs: {args.epochs} '
          f'| Folds: {args.folds} | Seeds: {args.seeds} | Device: {device} | torch_threads: {torch.get_num_threads()}')
    run_search(model_name=args.model, region=args.region, n_trials=args.n_trials, epochs=args.epochs,
               seeds=tuple(args.seeds), folds=tuple(args.folds), split_seed=args.split_seed, n_folds=args.n_folds,
               n_synthetic=args.n_synthetic, dataset_dir=args.dataset_dir, n_jobs=args.n_jobs,
               sampler_seed=args.sampler_seed, device=device, cv_mode=args.cv_mode, part_dropout=args.part_dropout)

if __name__ == '__main__':
    main()
