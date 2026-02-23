import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from data import BreathDataset, load_dataset, split_dataset
from models.vae import VAE, elbo_loss

# cli arguments
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train VAE on breath signals')
    p.add_argument('--region', required=True, choices=['mouth', 'nose'])
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--latent_dim', type=int, default=16)
    p.add_argument('--dataset_dir', default='dataset')
    p.add_argument('--log_every', type=int, default=10)
    return p.parse_args()

# training
def beta_schedule(epoch: int, total_epochs: int) -> float:
    """Linear beta anneal: 0 -> 1 over first half of training to avoid posterior collapse."""
    return min(1.0, epoch / (total_epochs * 0.5))

def train_one_epoch(model, loader, optimizer, epoch, total_epochs, device):
    model.train()
    beta = beta_schedule(epoch, total_epochs)
    total_loss = recon_sum = kl_sum = 0.0
    for signal, _time, _label in loader:
        signal = signal.to(device)
        x_hat, mu, logvar = model(signal)
        loss, recon, kl = elbo_loss(signal, x_hat, mu, logvar, beta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        recon_sum += recon.item()
        kl_sum += kl.item()
    n = len(loader)
    return total_loss / n, recon_sum / n, kl_sum / n

@torch.no_grad()
def evaluate(model, loader, epoch, total_epochs, device):
    model.eval()
    beta = beta_schedule(epoch, total_epochs)
    total_loss = recon_sum = kl_sum = 0.0
    for signal, _time, _label in loader:
        signal = signal.to(device)
        x_hat, mu, logvar = model(signal)
        loss, recon, kl = elbo_loss(signal, x_hat, mu, logvar, beta)
        total_loss += loss.item()
        recon_sum += recon.item()
        kl_sum += kl.item()
    n = len(loader)
    return total_loss / n, recon_sum / n, kl_sum / n

# plotting
def plot_results(model, train_ds, region, device, out_dir='results'):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    # --- 1. Reconstruction: 3 real vs reconstructed signals ---
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    fig.suptitle(f'VAE Reconstruction — {region}', fontsize=13)
    channel_names = ['Humidity (norm.)', 'Temperature (norm.)']

    sample_indices = np.random.default_rng(42).choice(len(train_ds), size=3, replace=False)
    for col, idx in enumerate(sample_indices):
        signal, time, label = train_ds[idx]
        signal_batch = signal.unsqueeze(0).to(device)
        with torch.no_grad():
            x_hat, _, _ = model(signal_batch)
        x_hat = x_hat.squeeze(0).cpu().numpy()
        signal_np = signal.numpy()
        time_np = time.numpy()

        for row in range(2):
            ax = axes[row, col]
            ax.plot(time_np, signal_np[row], label='real', color='steelblue')
            ax.plot(time_np, x_hat[row], label='recon', color='tomato', linestyle='--')
            if col == 0:
                ax.set_ylabel(channel_names[row])
            if row == 0:
                ax.set_title(f'Sample {idx}')
                ax.legend(fontsize=8)
            ax.set_xlabel('time (s)')

    plt.tight_layout()
    path = os.path.join(out_dir, f'vae_{region}_reconstruction.png')
    plt.savefig(path, dpi=120)
    plt.close()
    print(f'Saved reconstruction plot → {path}')

    # --- 2. Generation: 9 random samples from prior ---
    samples = model.sample(9, device).cpu().numpy()
    # Approximate time axis from first training sample
    time_np = train_ds[0][1].numpy()

    fig, axes = plt.subplots(3, 3, figsize=(12, 8))
    fig.suptitle(f'VAE Generated Samples — {region}', fontsize=13)
    for i, ax in enumerate(axes.flatten()):
        ax.plot(time_np, samples[i, 0], label='humidity', color='steelblue')
        ax.plot(time_np, samples[i, 1], label='temp.', color='darkorange')
        ax.set_xlabel('time (s)')
        if i == 0:
            ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(out_dir, f'vae_{region}_samples.png')
    plt.savefig(path, dpi=120)
    plt.close()
    print(f'Saved generation plot → {path}')

# main 
def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Region: {args.region}  |  Device: {device}  |  Epochs: {args.epochs}')

    # Data
    df = load_dataset(args.dataset_dir)
    df = df[df['region'] == args.region].reset_index(drop=True)
    print(f'Trials for {args.region}: {len(df)}')

    df_train, df_val, df_test = split_dataset(df, random_state=42)
    train_ds = BreathDataset(df_train)
    val_ds   = BreathDataset(df_val,  stats=train_ds.stats)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    # Model
    model = VAE(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    os.makedirs('checkpoints', exist_ok=True)
    ckpt_path = f'checkpoints/vae_{args.region}.pt'
    best_val_loss = math.inf

    for epoch in range(1, args.epochs + 1):
        train_loss, train_recon, train_kl = train_one_epoch(
            model, train_loader, optimizer, epoch, args.epochs, device)
        val_loss, val_recon, val_kl = evaluate(
            model, val_loader, epoch, args.epochs, device)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'epoch': epoch,
                        'model_state': model.state_dict(),
                        'stats': train_ds.stats,
                        'latent_dim': args.latent_dim,
                        'region': args.region}, ckpt_path)

        if epoch % args.log_every == 0 or epoch == 1:
            beta = beta_schedule(epoch, args.epochs)
            print(f'Epoch {epoch:4d}/{args.epochs} | β={beta:.2f} | '
                  f'train loss={train_loss:.4f} (recon={train_recon:.4f}, kl={train_kl:.4f}) | '
                  f'val loss={val_loss:.4f} (recon={val_recon:.4f}, kl={val_kl:.4f})')

    print(f'\nBest val loss: {best_val_loss:.4f}  |  Checkpoint: {ckpt_path}')

    # Load best model for evaluation plots
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    plot_results(model, train_ds, args.region, device)

if __name__ == '__main__':
    main()
