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

from core.data import BreathDataset, load_dataset, split_dataset
from core.train import evaluate, train_vae_one_epoch
from core.tstr import evaluate_classifier, extract_fixed_features, generate_synthetic_signals, load_cache, train_stacking_classifier, trtr
from core.utils import make_generator, seed_everything, seed_worker
from models.vae import CVAE, VAE

def _empty_device_cache(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()

def _train_model(model_name: str, params: dict, region: str, dataset_dir: str, epochs: int, seed: int, device: torch.device) -> tuple[torch.nn.Module, dict]:
    seed_everything(seed)
    g = make_generator(seed)

    df = load_dataset(dataset_dir)
    df = df[df['region'] == region].reset_index(drop=True)
    df_train, df_val, _ = split_dataset(df, random_state=seed)
    train_ds = BreathDataset(df_train)
    val_ds = BreathDataset(df_val, stats=train_ds.stats)
    train_loader = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True, drop_last=False, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_ds, batch_size=params['batch_size'], shuffle=False)

    if model_name == 'cvae_part':
        model = CVAE(latent_dim=params['latent_dim'], embed_dim=params['embed_dim'],
                     condition_on_participant=True, part_embed_dim=params['part_embed_dim']).to(device)
    elif model_name == 'cvae':
        model = CVAE(latent_dim=params['latent_dim'], embed_dim=params['embed_dim']).to(device)
    else:
        model = VAE(latent_dim=params['latent_dim']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    conditional = model_name in ('cvae', 'cvae_part')
    use_participant = model_name == 'cvae_part'
    for epoch in range(1, epochs + 1):
        train_vae_one_epoch(model, train_loader, optimizer, epoch, epochs, device, params['free_bits'], conditional=conditional, use_participant=use_participant, beta_max=params['beta_max'])
        evaluate(model, val_loader, epoch, epochs, device, params['free_bits'], conditional=conditional, use_participant=use_participant, beta_max=params['beta_max'])
        scheduler.step()

    return model, train_ds.stats

def _tstr_accuracy(model: torch.nn.Module, model_name: str, region: str, dataset_dir: str, n_jobs: int, seed: int, n_synthetic: int | None, device: torch.device) -> float:
    cache = load_cache(region, seed)
    if cache is None:
        cache = trtr(dataset_dir, region, n_jobs, seed)

    n_synth = n_synthetic if n_synthetic is not None else cache['n_train']
    synth_signals, synth_labels = generate_synthetic_signals(model, model_name, n_synth, cache['stats'], device)

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

def make_objective(model_name: str, region: str, dataset_dir: str = 'dataset', epochs: int = 500, seeds: tuple[int, ...] = (42,), n_synthetic: int | None = None, n_jobs: int = 4, device: torch.device | None = None):
    device = device or torch.device('cpu')

    def objective(trial: optuna.Trial) -> float:
        params = {'latent_dim': trial.suggest_categorical('latent_dim', [8, 16, 32, 64]),
                  'free_bits': trial.suggest_float('free_bits', 0.0, 2.0),
                  'beta_max': trial.suggest_float('beta_max', 1e-2, 1.0, log=True),
                  'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
                  'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
                  'beta_warmup_epochs': trial.suggest_int('beta_warmup_epochs', 50, 300)}
        if model_name in ('cvae', 'cvae_part'):
            params['embed_dim'] = trial.suggest_categorical('embed_dim', [4, 8, 16, 32])
        if model_name == 'cvae_part':
            params['part_embed_dim'] = trial.suggest_categorical('part_embed_dim', [4, 8, 16, 32])

        accs = []
        model = None
        try:
            for i, seed in enumerate(seeds):
                try:
                    model, _stats = _train_model(model_name, params, region, dataset_dir, epochs, seed, device)
                    acc = _tstr_accuracy(model, model_name, region, dataset_dir, n_jobs, seed, n_synthetic, device)
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

def run_search(model_name: str, region: str, n_trials: int = 50, epochs: int = 500, seeds: tuple[int, ...] = (42,), n_synthetic: int | None = None, dataset_dir: str = 'dataset', n_jobs: int = 4, sampler_seed: int = 42, device: torch.device | None = None) -> optuna.Study:
    os.makedirs('results/tuning', exist_ok=True)
    storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(f'results/tuning/{model_name}_{region}_v3.log'))
    sampler = optuna.samplers.TPESampler(seed=sampler_seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2)
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner,
                                study_name=f'{model_name}_tstr_{region}',
                                storage=storage, load_if_exists=True)
    objective = make_objective(model_name, region, dataset_dir, epochs, seeds, n_synthetic, n_jobs, device)
    study.optimize(objective, n_trials=n_trials, callbacks=[_log_callback], show_progress_bar=False)

    trials_path = f'results/tuning/{model_name}_{region}_trials.csv'
    best_path = f'results/tuning/{model_name}_{region}_best.json'
    study.trials_dataframe().to_csv(trials_path, index=False)
    with open(best_path, 'w') as f:
        json.dump({'model': model_name, 'region': region, 'seeds': list(seeds), 'epochs': epochs,
                   'best_accuracy': study.best_value, 'best_params': study.best_params}, f, indent=2)
    print(f'[TUNE] Best TSTR accuracy: {study.best_value:.4f}')
    print(f'[TUNE] Best params: {study.best_params}')
    print(f'[TUNE] Trials saved to {trials_path}\n[TUNE] Best params saved to {best_path}')
    return study

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, choices=['vae', 'cvae', 'cvae_part'])
    p.add_argument('--region', required=True, choices=['mouth', 'nose'])
    p.add_argument('--n_trials', type=int, default=50)
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 42, 7, 123])
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

    print(f'[TUNE] Model: {args.model} | Region: {args.region} | Trials: {args.n_trials} | Epochs: {args.epochs} '
          f'| Seeds: {args.seeds} | Device: {device} | torch_threads: {torch.get_num_threads()}')
    run_search(model_name=args.model, region=args.region, n_trials=args.n_trials, epochs=args.epochs,
               seeds=tuple(args.seeds), n_synthetic=args.n_synthetic,
               dataset_dir=args.dataset_dir, n_jobs=args.n_jobs,
               sampler_seed=args.sampler_seed, device=device)

if __name__ == '__main__':
    main()
