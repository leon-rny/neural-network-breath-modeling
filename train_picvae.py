import argparse
import csv
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from core.data import BreathDataset, load_dataset, split_dataset
from models.vae import PICVAE, elbo_loss

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--region', required=True, choices=['mouth', 'nose'])
    p.add_argument('--dataset_dir', default='dataset')
    p.add_argument('--log_every', type=int, default=25)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--latent_dim', type=int, default=32)
    p.add_argument('--embed_dim', type=int, default=16)
    p.add_argument('--free_bits', type=float, default=0.0)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--lambda_physics', type=float, default=0.1)
    p.add_argument('--ode_params_path', default='results/ode_fit/ode_params_all.csv')
    return p.parse_args()

def beta_capped(epoch: int, total_epochs: int, beta_max: float = 0.1) -> float:
    """Linear warmup from 0 → beta_max over the first half of training, then hold."""
    warmup_epochs = total_epochs * 0.5
    return min(beta_max, (epoch / warmup_epochs) * beta_max)

def train_one_epoch(model: PICVAE, loader: DataLoader, optimizer: torch.optim.Optimizer, epoch: int, total_epochs: int, device: torch.device, free_bits: float, lambda_physics: float) -> tuple[float, float, float, float]:
    model.train()
    beta = beta_capped(epoch, total_epochs)
    total_loss = recon_sum = kl_sum = phys_sum = 0.0
    for signal, _time, label, _participant in loader:
        signal = signal.to(device)
        label = label.long().to(device)
        x_hat, mu, logvar = model(signal, label)
        elbo, recon, kl = elbo_loss(signal, x_hat, mu, logvar, beta, free_bits)
        phys = model.physics_loss(x_hat, label)
        loss = elbo + lambda_physics * phys
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        recon_sum  += recon.item()
        kl_sum     += kl.item()
        phys_sum   += phys.item()
    n = len(loader)
    return total_loss / n, recon_sum / n, kl_sum / n, phys_sum / n


@torch.no_grad()
def evaluate(
    model: PICVAE,
    loader: DataLoader,
    epoch: int,
    total_epochs: int,
    device: torch.device,
    free_bits: float,
    lambda_physics: float,
) -> tuple[float, float, float, float]:
    model.eval()
    beta = beta_capped(epoch, total_epochs)
    total_loss = recon_sum = kl_sum = phys_sum = 0.0
    for signal, _time, label, _participant in loader:
        signal = signal.to(device)
        label = label.long().to(device)
        x_hat, mu, logvar = model(signal, label)
        elbo, recon, kl = elbo_loss(signal, x_hat, mu, logvar, beta, free_bits)
        phys = model.physics_loss(x_hat, label)
        total_loss += (elbo + lambda_physics * phys).item()
        recon_sum  += recon.item()
        kl_sum     += kl.item()
        phys_sum   += phys.item()
    n = len(loader)
    return total_loss / n, recon_sum / n, kl_sum / n, phys_sum / n


@torch.no_grad()
def active_dims(model: PICVAE, dataset: BreathDataset, device: torch.device, threshold: float = 0.1) -> int:
    mus, logvars = [], []
    for i in range(len(dataset)):
        signal, _, label, _ = dataset[i]
        signal = signal.unsqueeze(0).to(device)
        mu, logvar = model.encoder(signal, torch.tensor([label]).long().to(device))
        mus.append(mu.squeeze(0).cpu())
        logvars.append(logvar.squeeze(0).cpu())
    mus = torch.stack(mus)
    logvars = torch.stack(logvars)
    kl_per_dim = -0.5 * (1 + logvars - mus.pow(2) - logvars.exp()).mean(dim=0)
    return int((kl_per_dim > threshold).sum().item())


def main() -> None:
    args = parse_args()

    os.makedirs('results/picvae', exist_ok=True)
    run_id = (
        f'{args.region}_s{args.seed}_ld{args.latent_dim}'
        f'_ed{args.embed_dim}_fb{args.free_bits}_lp{args.lambda_physics}'
    )
    ckpt_path    = f'results/picvae/{run_id}_checkpoint.pt'
    history_path = f'results/picvae/{run_id}_train_history.csv'

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps'  if torch.backends.mps.is_available() else 'cpu'
    )
    print(
        f'Model: picvae | Region: {args.region} | Device: {device} | '
        f'Epochs: {args.epochs} | lambda_physics: {args.lambda_physics}'
    )

    df = load_dataset(args.dataset_dir)
    df = df[df['region'] == args.region].reset_index(drop=True)
    df_train, df_val, _ = split_dataset(df, random_state=args.seed)
    train_ds = BreathDataset(df_train)
    val_ds   = BreathDataset(df_val, stats=train_ds.stats)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)
    print(f'[TRAIN] Trials for {args.region}: {len(df)}')

    model = PICVAE(
        latent_dim=args.latent_dim,
        embed_dim=args.embed_dim,
        region=args.region,
        ode_params_path=args.ode_params_path,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = torch.inf
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_recon, train_kl, train_phys = train_one_epoch(
            model, train_loader, optimizer, epoch, args.epochs,
            device, args.free_bits, args.lambda_physics,
        )
        val_loss, val_recon, val_kl, val_phys = evaluate(
            model, val_loader, epoch, args.epochs,
            device, args.free_bits, args.lambda_physics,
        )
        scheduler.step()

        if val_loss < best_val_loss and beta_capped(epoch, args.epochs) >= 1.0:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'stats': train_ds.stats,
                'latent_dim': args.latent_dim,
                'embed_dim': args.embed_dim,
                'region': args.region,
                'lambda_physics': args.lambda_physics,
            }, ckpt_path)

        beta = beta_capped(epoch, args.epochs)
        n_active = (
            active_dims(model, train_ds, device)
            if epoch % args.log_every == 0 or epoch == 1
            else (history[-1]['active_dims'] if history else 0)
        )
        history.append({
            'epoch': epoch, 'beta': beta,
            'train_loss': train_loss, 'train_recon': train_recon,
            'train_kl': train_kl,    'train_phys': train_phys,
            'val_loss': val_loss,    'val_recon': val_recon,
            'val_kl': val_kl,        'val_phys': val_phys,
            'active_dims': n_active,
        })
        if epoch % args.log_every == 0 or epoch == 1:
            print(
                f'[TRAIN] Epoch {epoch:4d}/{args.epochs} | beta={beta:.2f} | '
                f'train loss={train_loss:.4f} '
                f'(recon={train_recon:.4f}, kl={train_kl:.4f}, phys={train_phys:.4f}) | '
                f'val loss={val_loss:.4f} '
                f'(recon={val_recon:.4f}, kl={val_kl:.4f}, phys={val_phys:.4f}) | '
                f'active_dims={n_active}'
            )

    print(f'[TRAIN] Best val loss: {best_val_loss} | Checkpoint: {ckpt_path}')

    with open(history_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(f'[TRAIN] History saved to {history_path}')


if __name__ == '__main__':
    main()
