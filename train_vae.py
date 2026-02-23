import argparse
import csv
import os

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

# main 
def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Region: {args.region} | Device: {device} | Epochs: {args.epochs}')

    # data
    df = load_dataset(args.dataset_dir)
    df = df[df['region'] == args.region].reset_index(drop=True)
    print(f'Trials for {args.region}: {len(df)}')

    df_train, df_val, df_test = split_dataset(df, random_state=42)
    train_ds = BreathDataset(df_train)
    val_ds = BreathDataset(df_val,  stats=train_ds.stats)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    # model
    model = VAE(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # training loop
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results/vae', exist_ok=True)
    ckpt_path = f'checkpoints/vae_{args.region}.pt'
    history_path = f'results/vae/train_{args.region}.csv'
    best_val_loss = torch.inf
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_recon, train_kl = train_one_epoch(
            model, train_loader, optimizer, epoch, args.epochs, device)
        val_loss, val_recon, val_kl = evaluate(
            model, val_loader, epoch, args.epochs, device)
        scheduler.step()

        # save best model
        if val_loss < best_val_loss and beta_schedule(epoch, args.epochs) >= 1.0:
            best_val_loss = val_loss
            torch.save({'epoch': epoch,
                        'model_state': model.state_dict(),
                        'stats': train_ds.stats,
                        'latent_dim': args.latent_dim,
                        'region': args.region}, ckpt_path)

        beta = beta_schedule(epoch, args.epochs)

        # logging
        history.append({'epoch': epoch, 'beta': beta,
                        'train_loss': train_loss, 'train_recon': train_recon, 'train_kl': train_kl,
                        'val_loss': val_loss,   'val_recon': val_recon,   'val_kl': val_kl})
        if epoch % args.log_every == 0 or epoch == 1:
            print(f'Epoch {epoch:4d}/{args.epochs} | β={beta:.2f} | '
                  f'train loss={train_loss:.4f} (recon={train_recon:.4f}, kl={train_kl:.4f}) | '
                  f'val loss={val_loss:.4f} (recon={val_recon:.4f}, kl={val_kl:.4f})')

    # results
    with open(history_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(f'\nBest val loss: {best_val_loss:.4f} | Checkpoint: {ckpt_path}')
    print(f'Training history saved to {history_path}')

if __name__ == '__main__':
    main()
