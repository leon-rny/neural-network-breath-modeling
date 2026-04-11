import argparse
import csv
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from core.data import CLASSES, BreathDataset, load_dataset, split_dataset
from core.train import active_dims
from core.tstr import (evaluate_classifier, extract_fixed_features,
                       load_cache, train_stacking_classifier, trtr)
from models.vae import CVAE, elbo_loss

# jittering augmentation wrapper
class JitteredDataset(Dataset):
    def __init__(self, base: BreathDataset, n_copies: int, alpha: float) -> None:
        self.base = base
        self.n_copies = n_copies
        self.alpha = alpha
        signals = torch.stack([base[i][0] for i in range(len(base))])
        self.std_per_ch = signals.std(dim=(0, 2)).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.base) * self.n_copies

    def __getitem__(self, idx: int):
        base_idx = idx % len(self.base)
        signal, time, label, participant = self.base[base_idx]
        noise = self.alpha * self.std_per_ch * torch.randn_like(signal)
        return signal + noise, time, label, participant

# configs
CONFIGS: dict[str, dict] = {'baseline': {'alpha': 0.0, 'n_copies': 0},
                            'a0.01_n2': {'alpha': 0.01, 'n_copies': 2},
                            'a0.01_n5': {'alpha': 0.01, 'n_copies': 5},
                            'a0.01_n10': {'alpha': 0.01, 'n_copies': 10},
                            'a0.05_n2': {'alpha': 0.05, 'n_copies': 2},
                            'a0.05_n5': {'alpha': 0.05, 'n_copies': 5},
                            'a0.05_n10': {'alpha': 0.05, 'n_copies': 10},
                            'a0.1_n2': {'alpha': 0.1, 'n_copies': 2},
                            'a0.1_n5': {'alpha': 0.1, 'n_copies': 5},
                            'a0.1_n10': {'alpha': 0.1, 'n_copies': 10}}

BETA_MAX = {'mouth': 0.1, 'nose': 0.01}
LATENT_DIM = 16
EMBED_DIM = 8
PART_EMBED_DIM = 8
BATCH_SIZE = 32
LOG_EVERY = 25
SAVE_WINDOW_START = 400
RESULTS_DIR = 'results/ablation_cvae_jittering'

# cli
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='CVAE jittering augmentation ablation')
    # sweep control
    p.add_argument('--configs', default=','.join(CONFIGS))
    p.add_argument('--config', default=None)
    p.add_argument('--alpha', type=float, default=None)
    p.add_argument('--n_copies', type=int, default=None)
    # common
    p.add_argument('--seeds', default='0,1,7,42,123')
    p.add_argument('--regions', default='mouth,nose')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--region', default=None)
    p.add_argument('--dataset_dir', default='dataset')
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--n_jobs', type=int, default=4)
    p.add_argument('--skip_existing', action='store_true')
    return p.parse_args()

def beta_capped(epoch: int, total_epochs: int, beta_max: float) -> float:
    """Linear warmup from 0 → beta_max over the first half of training, then hold."""
    warmup_epochs = total_epochs * 0.5
    return min(beta_max, (epoch / warmup_epochs) * beta_max)

# paths
def ckpt_path(config: str, region: str, seed: int) -> str:
    return f'{RESULTS_DIR}/{config}_{region}_s{seed}.pt'

def hist_path(config: str, region: str, seed: int) -> str:
    return f'{RESULTS_DIR}/{config}_{region}_s{seed}_history.csv'

# training
def train_config(config: str, region: str, seed: int, dataset_dir: str, device: torch.device, epochs: int, lr: float) -> None:
    cfg = CONFIGS[config]
    alpha = cfg['alpha']
    n_copies = cfg['n_copies']
    beta_max = BETA_MAX[region]

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # data
    df = load_dataset(dataset_dir)
    df = df[df['region'] == region].reset_index(drop=True)
    df_train, df_val, _ = split_dataset(df, random_state=seed)
    train_ds = BreathDataset(df_train)
    val_ds = BreathDataset(df_val, stats=train_ds.stats)

    # augmentation (training only)
    if alpha > 0 and n_copies > 0:
        aug_ds = JitteredDataset(train_ds, n_copies, alpha)
        train_loader = DataLoader(aug_ds, batch_size=BATCH_SIZE,
                                  shuffle=True, drop_last=False)
        print(f'[Jitter] {len(train_ds)}: {len(aug_ds)} samples (alpha={alpha}, n_copies={n_copies})')
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                  shuffle=True, drop_last=False)

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # model (conv_baseline with participant conditioning)
    model = CVAE(latent_dim=LATENT_DIM, embed_dim=EMBED_DIM,
                 condition_on_participant=True,
                 part_embed_dim=PART_EMBED_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        beta = beta_capped(epoch, epochs, beta_max)

        # train
        model.train()
        t_loss = t_recon = t_kl = 0.0
        n_batches = 0
        for signal, _time, label, participant in train_loader:
            signal = signal.to(device)
            label = label.long().to(device)
            participant = participant.long().to(device)

            optimizer.zero_grad()
            x_hat, mu, logvar = model(signal, label, participant)
            loss, recon, kl = elbo_loss(signal, x_hat, mu, logvar, beta)
            loss.backward()
            optimizer.step()

            t_loss += loss.item()
            t_recon += recon.item()
            t_kl += kl.item()
            n_batches += 1
        t_loss /= n_batches
        t_recon /= n_batches
        t_kl /= n_batches

        # validate
        model.eval()
        v_loss = v_recon = v_kl = 0.0
        with torch.no_grad():
            for signal, _time, label, participant in val_loader:
                signal = signal.to(device)
                label = label.long().to(device)
                participant = participant.long().to(device)
                x_hat, mu, logvar = model(signal, label, participant)
                loss, recon, kl = elbo_loss(signal, x_hat, mu, logvar, beta)
                v_loss += loss.item()
                v_recon += recon.item()
                v_kl += kl.item()
        n_val   = len(val_loader)
        v_loss  /= n_val
        v_recon /= n_val
        v_kl    /= n_val

        scheduler.step()

        # checkpoint
        if epoch >= SAVE_WINDOW_START and v_loss < best_val_loss:
            best_val_loss = v_loss
            os.makedirs(RESULTS_DIR, exist_ok=True)
            torch.save({'config': config,
                        'region': region,
                        'seed': seed,
                        'epoch': epoch,
                        'model_state': model.state_dict(),
                        'stats': train_ds.stats,
                        'latent_dim': LATENT_DIM,
                        'embed_dim': EMBED_DIM,
                        'part_embed_dim': PART_EMBED_DIM,
                        'beta_max': beta_max,
                        'alpha': alpha,
                        'n_copies': n_copies}, ckpt_path(config, region, seed))

        # active dims
        if epoch % LOG_EVERY == 0 or epoch == 1:
            n_active = active_dims(model, train_ds, device,
                                   conditional=True, use_participant=True)
        else:
            n_active = history[-1]['active_dims'] if history else 0

        history.append({'epoch': epoch,
                        'beta': beta,
                        'train_loss': t_loss, 'train_recon': t_recon, 'train_kl': t_kl,
                        'val_loss': v_loss, 'val_recon': v_recon, 'val_kl': v_kl,
                        'active_dims': n_active})

        if epoch % LOG_EVERY == 0 or epoch == 1:
            print(f'  [{config}|{region}|s{seed}] epoch {epoch:4d}/{epochs} | β={beta:.5f} | train {t_loss:.4f} (r={t_recon:.4f}, kl={t_kl:.4f}) | val {v_loss:.4f} (r={v_recon:.4f}, kl={v_kl:.4f}) | active_dims={n_active}')

    # persist full training history
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(hist_path(config, region, seed), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    print(f'  [{config}|{region}|s{seed}] done. best_val={best_val_loss:.4f} | ckpt: {ckpt_path(config, region, seed)}')

# tstr evaluation
def eval_config(config: str, region: str, seed: int, cache: dict, device: torch.device, n_jobs: int) -> dict:
    path = ckpt_path(config, region, seed)
    if not os.path.exists(path):
        raise FileNotFoundError(f'Checkpoint not found: {path}')

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = CVAE(latent_dim=ckpt['latent_dim'],
                 embed_dim=ckpt['embed_dim'],
                 condition_on_participant=True,
                 part_embed_dim=ckpt['part_embed_dim']).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    stats = ckpt['stats']

    # generate synthetic signals
    n_synthetic = cache['n_train']
    n_per_class = n_synthetic // len(CLASSES)
    remainder = n_synthetic % len(CLASSES)
    counts = [n_per_class + (1 if i < remainder else 0) for i in range(len(CLASSES))]

    mean_t = torch.tensor(stats['mean'], dtype=torch.float32).view(1, 2, 1).to(device)
    std_t  = torch.tensor(stats['std'],  dtype=torch.float32).view(1, 2, 1).to(device)

    all_signals, all_labels = [], []
    for cls_idx, count in enumerate(counts):
        y_cls = torch.tensor(cls_idx, dtype=torch.long)
        sigs  = model.sample(count, y_cls, device)
        all_signals.append((sigs * std_t + mean_t).cpu().numpy())
        all_labels.append(np.full(count, cls_idx))
    synth_signals = np.concatenate(all_signals, axis=0)
    synth_labels  = np.concatenate(all_labels,  axis=0)
    print(f'  [{config}|{region}|s{seed}] generated {n_synthetic} synthetic signals')

    # tsfresh features using TRTR cache's top-20
    n, _C, T = synth_signals.shape
    df_long = pd.DataFrame({'id': np.repeat(np.arange(n), T),
                            'time': np.tile(np.arange(T), n),
                            'Humidity': synth_signals[:, 0, :].ravel(),
                            'Temperature': synth_signals[:, 1, :].ravel()})
    X_raw = extract_fixed_features(df_long, cache['top_20_features_raw'], n_jobs)
    X_san = X_raw.copy()
    X_san.columns = [re.sub(r'[^\w]', '_', c) for c in X_san.columns]
    X_synth = X_san[cache['top_20_features_sanitized']].values

    stacker = train_stacking_classifier(X_synth, synth_labels, seed)
    metrics = evaluate_classifier(stacker, cache['X_test_top'], cache['y_test'])

    # final-epoch diagnostics from history
    kl_final = float('nan')
    active_dims_final = 0
    hp = hist_path(config, region, seed)
    if os.path.exists(hp):
        hist_df = pd.read_csv(hp)
        kl_final          = float(hist_df['train_kl'].iloc[-1])
        active_dims_final = int(hist_df['active_dims'].iloc[-1])

    return {'config': config,
            'alpha': CONFIGS[config]['alpha'],
            'n_copies': CONFIGS[config]['n_copies'],
            'region': region,
            'seed': seed,
            'accuracy': metrics['accuracy'],
            'f1_weighted': metrics['f1_weighted'],
            'roc_auc': metrics['roc_auc_ovr'],
            'active_dims': active_dims_final,
            'kl_final': kl_final,
            'f1_brady': metrics['per_class_f1']['bradypnea'],
            'f1_eupnea': metrics['per_class_f1']['eupnea'],
            'f1_tachy': metrics['per_class_f1']['tachypnea']}

# results
def save_results(rows: list[dict]) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = f'{RESULTS_DIR}/summary.csv'
    df_new = pd.DataFrame(rows)
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        df_new = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates(
            subset=['config', 'region', 'seed'], keep='last')
    df_new.to_csv(csv_path, index=False)
    print(f'[JITTER] Saved {len(df_new)} rows → {csv_path}')

def print_summary(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    for region in sorted(df['region'].unique()):
        rdf = df[df['region'] == region]
        mean_acc = rdf.groupby('config')['accuracy'].mean()
        best_config = mean_acc.idxmax()
        # rank configs by mean accuracy descending
        ranked = mean_acc.sort_values(ascending=False).index.tolist()
        print(f'\nRegion: {region}  (β_max={BETA_MAX[region]})')
        print(f"  {'#':>2} {'Config':<14} {'alpha':>5} {'n_cp':>5} {'Accuracy':>14} {'F1-W':>14} {'ROC-AUC':>14} {'KL':>8} {'ActDims':>8}")
        for rank, cfg in enumerate(ranked, 1):
            vdf = rdf[rdf['config'] == cfg]
            a = CONFIGS[cfg]['alpha']
            nc = CONFIGS[cfg]['n_copies']
            acc = vdf['accuracy'].mean()
            acc_s = vdf['accuracy'].std()
            f1 = vdf['f1_weighted'].mean()
            f1_s = vdf['f1_weighted'].std()
            roc = vdf['roc_auc'].mean()
            roc_s = vdf['roc_auc'].std()
            kl = vdf['kl_final'].mean()
            ad = vdf['active_dims'].mean()
            flag = ' *' if cfg == best_config else ''
            print(f"  {rank:>2} {cfg:<14} {a:>5.2f} {nc:>5d} {acc:.3f}±{acc_s:.3f}  {f1:.3f}±{f1_s:.3f}  {roc:.3f}±{roc_s:.3f}  {kl:>8.4f}  {ad:>8.1f}{flag}")
    print(' * = best mean accuracy for that region')

# main
def main() -> None:
    args = parse_args()

    # resolve config list
    if args.alpha is not None and args.n_copies is not None:
        if args.alpha == 0:
            configs = ['baseline']
        else:
            name = f'a{args.alpha}_n{args.n_copies}'
            if name not in CONFIGS:
                CONFIGS[name] = {'alpha': args.alpha, 'n_copies': args.n_copies}
            configs = [name]
    elif args.config is not None:
        configs = [args.config]
    else:
        configs = args.configs.split(',')

    seeds = [args.seed] if args.seed is not None else [int(s) for s in args.seeds.split(',')]
    regions = [args.region] if args.region else args.regions.split(',')

    for c in configs:
        if c not in CONFIGS:
            print(f'Unknown config "{c}". Valid: {list(CONFIGS)}')
            sys.exit(1)
    for r in regions:
        if r not in ('mouth', 'nose'):
            print(f'Unknown region "{r}". Valid: mouth, nose')
            sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'[JITTER] device={device} | configs={configs} | seeds={seeds} | regions={regions}')

    # main loop
    all_results: list[dict] = []

    for region in regions:
        # TRTR cache
        caches: dict[int, dict] = {}
        for seed in seeds:
            cache = load_cache(region, seed)
            if cache is None:
                print(f'[JITTER] Building TRTR cache region={region}, seed={seed} ...')
                cache = trtr(args.dataset_dir, region, args.n_jobs, seed)
            caches[seed] = cache
            print(f'[JITTER] TRTR cache ready: region={region}, seed={seed}, n_train={cache["n_train"]}')

        for config in configs:
            for seed in seeds:
                print(f'\n[JITTER] {config} | {region} | seed={seed}')

                # train
                path = ckpt_path(config, region, seed)
                if args.skip_existing and os.path.exists(path):
                    print('checkpoint exists, skipping training')
                else:
                    train_config(config, region, seed, args.dataset_dir,
                                 device, args.epochs, args.lr)

                # evaluate
                row = eval_config(config, region, seed, caches[seed],
                                  device, args.n_jobs)
                all_results.append(row)
                print(f'  TSTR: acc={row["accuracy"]:.3f}, f1={row["f1_weighted"]:.3f}, roc={row["roc_auc"]:.3f}, kl={row["kl_final"]:.4f}, active_dims={row["active_dims"]}')

    if all_results:
        save_results(all_results)
        print_summary(all_results)

if __name__ == '__main__':
    main()
