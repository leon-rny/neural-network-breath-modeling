import argparse
import csv
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from core.data import CLASSES, BreathDataset, load_dataset, split_dataset
from core.train import active_dims, beta_schedule
from core.tstr import evaluate_classifier, extract_fixed_features, load_cache, train_stacking_classifier, trtr
from ablations.models.cvae_ablation import VARIANT_MAP
from models.vae import elbo_loss

# fix variables expect architecture
LATENT_DIM = 16
EMBED_DIM = 8
PART_EMBED_DIM = 8
EPOCHS = 500
BATCH_SIZE = 32
LR = 1e-3
LOG_EVERY = 25

# cli
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='CVAE architecture ablation study')
    p.add_argument('--variants', default=','.join(VARIANT_MAP))
    p.add_argument('--seeds', default='42,43,44,45,46')
    p.add_argument('--regions', default='mouth,nose')
    p.add_argument('--variant', default=None,)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--region',  default=None,)
    p.add_argument('--dataset_dir', default='dataset')
    p.add_argument('--n_jobs', type=int, default=4)
    p.add_argument('--skip_existing', action='store_true')
    return p.parse_args()

# paths
def ckpt_path(variant: str, region: str, seed: int) -> str:
    return f'results/ablation_cvae_architecture/{variant}_{region}_s{seed}.pt'

def hist_path(variant: str, region: str, seed: int) -> str:
    return f'results/ablation_cvae_architecture/{variant}_{region}_s{seed}_history.csv'

# training
def train_variant(variant: str, region: str, seed: int, dataset_dir: str, device: torch.device) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # data
    df = load_dataset(dataset_dir)
    df = df[df['region'] == region].reset_index(drop=True)
    df_train, df_val, _ = split_dataset(df, random_state=seed)
    train_ds = BreathDataset(df_train)
    val_ds = BreathDataset(df_val, stats=train_ds.stats)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # model + optimiser
    ModelClass = VARIANT_MAP[variant]
    model = ModelClass(latent_dim=LATENT_DIM, embed_dim=EMBED_DIM, condition_on_participant=True, part_embed_dim=PART_EMBED_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float('inf')
    history = []

    for epoch in range(1, EPOCHS + 1):
        beta = beta_schedule(epoch, EPOCHS)

        # train
        model.train()
        t_loss = t_recon = t_kl = 0.0
        for signal, _time, label, participant in train_loader:
            signal = signal.to(device)
            label = label.long().to(device)
            participant = participant.long().to(device)
            x_hat, mu, logvar = model(signal, label, participant)
            loss, recon, kl = elbo_loss(signal, x_hat, mu, logvar, beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            t_recon += recon.item()
            t_kl += kl.item()
        n_tr = len(train_loader)
        t_loss /= n_tr
        t_recon /= n_tr
        t_kl /= n_tr

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
        n_val = len(val_loader)
        v_loss /= n_val
        v_recon /= n_val
        v_kl /= n_val

        scheduler.step()

        # checkpoint only once beta has fully warmed up
        if v_loss < best_val_loss and beta >= 1.0:
            best_val_loss = v_loss
            os.makedirs('results/ablation', exist_ok=True)
            torch.save({'variant':    variant,
                        'region': region,
                        'seed': seed,
                        'epoch': epoch,
                        'model_state': model.state_dict(),
                        'stats': train_ds.stats,
                        'latent_dim': LATENT_DIM,
                        'embed_dim': EMBED_DIM,
                        'part_embed_dim': PART_EMBED_DIM}, ckpt_path(variant, region, seed))

        # active dims
        if epoch % LOG_EVERY == 0 or epoch == 1:
            n_active = active_dims(model, train_ds, device,
                                   conditional=True, use_participant=True)
        else:
            n_active = history[-1]['active_dims'] if history else 0

        history.append({'epoch': epoch, 'beta': beta,
                        'train_loss': t_loss, 'train_recon': t_recon, 'train_kl': t_kl,
                        'val_loss': v_loss, 'val_recon': v_recon, 'val_kl': v_kl,
                        'active_dims': n_active})

        if epoch % LOG_EVERY == 0 or epoch == 1:
            print(f'  [{variant}|{region}|s{seed}] epoch {epoch:4d}/{EPOCHS} | beta={beta:.2f} | train {t_loss:.4f} (r={t_recon:.4f}, kl={t_kl:.4f}) | val {v_loss:.4f} (r={v_recon:.4f}, kl={v_kl:.4f}) | active_dims={n_active}')

    # persist history
    os.makedirs('results/ablation', exist_ok=True)
    with open(hist_path(variant, region, seed), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    print(f'  [{variant}|{region}|s{seed}] done. best_val={best_val_loss:.4f} | ckpt: {ckpt_path(variant, region, seed)}')

# tstr evaluation
def eval_variant(variant: str, region: str, seed: int,
                 cache: dict, device: torch.device, n_jobs: int) -> dict:
    path = ckpt_path(variant, region, seed)
    if not os.path.exists(path):
        raise FileNotFoundError(f'Checkpoint not found: {path}')

    ckpt = torch.load(path, map_location=device, weights_only=False)
    ModelClass = VARIANT_MAP[variant]
    model = ModelClass(latent_dim=ckpt['latent_dim'],
                       embed_dim=ckpt['embed_dim'],
                       condition_on_participant=True,
                       part_embed_dim=ckpt['part_embed_dim']).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    stats = ckpt['stats']

    # generate synthetic signals
    n_synthetic = cache['n_train']
    n_per_class = n_synthetic // len(CLASSES)
    remainder   = n_synthetic % len(CLASSES)
    counts = [n_per_class + (1 if i < remainder else 0) for i in range(len(CLASSES))]

    mean_t = torch.tensor(stats['mean'], dtype=torch.float32).view(1, 2, 1).to(device)
    std_t  = torch.tensor(stats['std'], dtype=torch.float32).view(1, 2, 1).to(device)

    all_signals, all_labels = [], []
    for cls_idx, count in enumerate(counts):
        y_cls = torch.tensor(cls_idx, dtype=torch.long)
        sigs  = model.sample(count, y_cls, device)
        all_signals.append((sigs * std_t + mean_t).cpu().numpy())
        all_labels.append(np.full(count, cls_idx))
    synth_signals = np.concatenate(all_signals, axis=0)
    synth_labels  = np.concatenate(all_labels,  axis=0)
    print(f'  [{variant}|{region}|s{seed}] generated {n_synthetic} synthetic signals')

    # tsfresh feature extraction on synthetic data
    n, _C, T = synth_signals.shape
    df_long = pd.DataFrame({'id': np.repeat(np.arange(n), T),
                            'time': np.tile(np.arange(T), n),
                            'Humidity': synth_signals[:, 0, :].ravel(),
                            'Temperature': synth_signals[:, 1, :].ravel()})
    X_raw = extract_fixed_features(df_long, cache['top_20_features_raw'], n_jobs)
    X_san = X_raw.copy()
    X_san.columns = [re.sub(r'[^\w]', '_', c) for c in X_san.columns]
    X_synth = X_san[cache['top_20_features_sanitized']].values

    # train stacker on synthetic, test on real
    stacker = train_stacking_classifier(X_synth, synth_labels, seed)
    metrics = evaluate_classifier(stacker, cache['X_test_top'], cache['y_test'])

    # pull final-epoch kl / active_dims from saved history
    kl_final = float('nan')
    active_dims_final = 0
    hp = hist_path(variant, region, seed)
    if os.path.exists(hp):
        hist_df = pd.read_csv(hp)
        kl_final          = float(hist_df['train_kl'].iloc[-1])
        active_dims_final = int(hist_df['active_dims'].iloc[-1])

    return {'variant': variant,
            'region': region,
            'seed': seed,
            'accuracy': metrics['accuracy'],
            'f1_weighted': metrics['f1_weighted'],
            'roc_auc': metrics['roc_auc_ovr'],
            'kl_final': kl_final,
            'active_dims': active_dims_final,
            'f1_brady': metrics['per_class_f1']['bradypnea'],
            'f1_eupnea': metrics['per_class_f1']['eupnea'],
            'f1_tachy': metrics['per_class_f1']['tachypnea']}

# results
def save_results(rows: list[dict]) -> None:
    os.makedirs('results/ablation_cvae_architecture', exist_ok=True)
    csv_path = 'results/ablation_cvae_architecture/summary.csv'
    df_new = pd.DataFrame(rows)
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        df_new = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates(
            subset=['variant', 'region', 'seed'], keep='last')
    df_new.to_csv(csv_path, index=False)
    print(f'[ABLATION] Saved {len(df_new)} rows → {csv_path}')

def print_summary(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    for region in df['region'].unique():
        rdf = df[df['region'] == region]
        best_variant = rdf.groupby('variant')['accuracy'].mean().idxmax()
        print(f'\nRegion: {region}')
        print(f"  {'Variant':<16} {'Accuracy':>12} {'F1-W':>12} {'ROC-AUC':>12} {'KL':>8} {'ActDims':>8}")
        print('  ' + '-' * 72)
        for variant in VARIANT_MAP:
            vdf = rdf[rdf['variant'] == variant]
            if vdf.empty:
                continue
            acc = vdf['accuracy'].mean()
            acc_s = vdf['accuracy'].std()
            f1  = vdf['f1_weighted'].mean()
            f1_s  = vdf['f1_weighted'].std()
            roc = vdf['roc_auc'].mean()
            roc_s = vdf['roc_auc'].std()
            kl  = vdf['kl_final'].mean()
            ad  = vdf['active_dims'].mean()
            flag = ' *' if variant == best_variant else ''
            print(f"  {variant:<16} {acc:.3f}±{acc_s:.3f}  {f1:.3f}±{f1_s:.3f}  {roc:.3f}±{roc_s:.3f}  {kl:>8.3f}  {ad:>8.1f}{flag}")
    print('  * = best accuracy for that region')

# main
def main() -> None:
    args = parse_args()

    variants = [args.variant] if args.variant else args.variants.split(',')
    seeds = [args.seed] if args.seed is not None else [int(s) for s in args.seeds.split(',')]
    regions = [args.region] if args.region else args.regions.split(',')

    for v in variants:
        if v not in VARIANT_MAP:
            print(f'Unknown variant "{v}". Valid: {list(VARIANT_MAP)}')
            sys.exit(1)
    for r in regions:
        if r not in ('mouth', 'nose'):
            print(f'Unknown region "{r}". Valid: mouth, nose')
            sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps'  if torch.backends.mps.is_available() else 'cpu')
    print(f'[ABLATION] device={device} | variants={variants} | seeds={seeds} | regions={regions}')

    all_results: list[dict] = []

    for region in regions:
        caches: dict[int, dict] = {}
        for seed in seeds:
            cache = load_cache(region, seed)
            if cache is None:
                print(f'[ABLATION] Building TRTR cache region={region}, seed={seed} ...')
                cache = trtr(args.dataset_dir, region, args.n_jobs, seed)
            caches[seed] = cache
            print(f'[ABLATION] TRTR cache ready: region={region}, seed={seed}, n_train={cache["n_train"]}, top_20 features selected')

        for variant in variants:
            for seed in seeds:
                print(f'\n[ABLATION] {variant} | {region} | seed={seed}')

                # train
                path = ckpt_path(variant, region, seed)
                if args.skip_existing and os.path.exists(path):
                    print('checkpoint exists, skipping training')
                else:
                    train_variant(variant, region, seed, args.dataset_dir, device)

                # evaluate
                row = eval_variant(variant, region, seed, caches[seed], device, args.n_jobs)
                all_results.append(row)
                print(f'  TSTR: acc={row["accuracy"]:.3f}, f1={row["f1_weighted"]:.3f}, roc={row["roc_auc"]:.3f}, kl={row["kl_final"]:.3f}, active_dims={row["active_dims"]}')

    if all_results:
        save_results(all_results)
        print_summary(all_results)

if __name__ == '__main__':
    main()
