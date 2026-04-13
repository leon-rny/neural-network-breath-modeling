import argparse
import csv
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.data import BreathDataset, load_dataset, split_dataset
from models.vae import VAE, CVAE, elbo_loss
from models.gan import CGAN, discriminator_loss, generator_loss, gradient_penalty
from models.pinn import BreathMLP

# cli arguments
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # general
    p.add_argument('--model', required=True, choices=['vae', 'cvae', 'cvae_part', 'gan', 'mlp'])
    p.add_argument('--region', required=True, choices=['mouth', 'nose'])
    p.add_argument('--dataset_dir', default='dataset')
    p.add_argument('--log_every', type=int, default=25)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--batch_size', type=int, default=32)
    # model-specific
    p.add_argument('--latent_dim', type=int, default=32)
    p.add_argument('--embed_dim', type=int, default=16)
    p.add_argument('--part_embed_dim', type=int, default=8)
    p.add_argument('--free_bits', type=float, default=0.0)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--lr_g', type=float, default=None)
    p.add_argument('--lr_d', type=float, default=None)
    p.add_argument('--n_critic', type=int, default=5)
    return p.parse_args()

# vae and cvae training
def beta_schedule(epoch: int, total_epochs: int) -> float:
    """Linear beta anneal: 0 -> 1 over first half of training to avoid posterior collapse."""
    return min(1.0, epoch / (total_epochs * 0.5))

def train_vae_one_epoch(model, loader, optimizer, epoch, total_epochs, device, free_bits, conditional=False, use_participant=False):
    model.train()
    beta = beta_schedule(epoch, total_epochs)
    total_loss = recon_sum = kl_sum = 0.0
    for signal, _time, label, participant in loader:
        signal = signal.to(device)
        if use_participant:
            label = label.long().to(device)
            participant = participant.long().to(device)
            x_hat, mu, logvar = model(signal, label, participant)
        elif conditional:
            label = label.long().to(device)
            x_hat, mu, logvar = model(signal, label)
        else:
            x_hat, mu, logvar = model(signal)
        loss, recon, kl = elbo_loss(signal, x_hat, mu, logvar, beta, free_bits)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        recon_sum  += recon.item()
        kl_sum     += kl.item()
    n = len(loader)
    return total_loss / n, recon_sum / n, kl_sum / n

@torch.no_grad()
def evaluate(model, loader, epoch, total_epochs, device, free_bits, conditional=False, use_participant=False):
    model.eval()
    beta = beta_schedule(epoch, total_epochs)
    total_loss = recon_sum = kl_sum = 0.0
    for signal, _time, label, participant in loader:
        signal = signal.to(device)
        if use_participant:
            label = label.long().to(device)
            participant = participant.long().to(device)
            x_hat, mu, logvar = model(signal, label, participant)
        elif conditional:
            label = label.long().to(device)
            x_hat, mu, logvar = model(signal, label)
        else:
            x_hat, mu, logvar = model(signal)
        loss, recon, kl = elbo_loss(signal, x_hat, mu, logvar, beta, free_bits)
        total_loss += loss.item()
        recon_sum  += recon.item()
        kl_sum     += kl.item()
    n = len(loader)
    return total_loss / n, recon_sum / n, kl_sum / n

@torch.no_grad()
def active_dims(model, dataset, device, threshold=0.1, conditional=False, use_participant=False):
    mus, logvars = [], []
    for i in range(len(dataset)):
        signal, _, label, participant = dataset[i]
        signal = signal.unsqueeze(0).to(device)
        if use_participant:
            mu, logvar = model.encoder(signal, torch.tensor([label]).long().to(device), torch.tensor([participant]).long().to(device))
        elif conditional:
            mu, logvar = model.encoder(signal, torch.tensor([label]).long().to(device))
        else:
            mu, logvar = model.encoder(signal)
        mus.append(mu.squeeze(0).cpu())
        logvars.append(logvar.squeeze(0).cpu())
    mus     = torch.stack(mus)
    logvars = torch.stack(logvars)
    kl_per_dim = -0.5 * (1 + logvars - mus.pow(2) - logvars.exp()).mean(dim=0)
    return int((kl_per_dim > threshold).sum().item())

# gan training
GP_LAMBDA = 10 # gradient penalty coefficient
CLS_LAMBDA = 1.0 # auxiliary classifier loss weight

def train_gan_one_epoch(model, loader, opt_g, opt_d, device, n_critic):
    model.train()
    d_loss_sum = g_loss_sum = 0.0
    n = 0
    for signal, _time, label, _participant in loader:
        signal = signal.to(device)
        label = label.long().to(device)
        B = signal.size(0)

        # train critic n_critic times per generator update
        for _ in range(n_critic):
            z = torch.randn(B, model.latent_dim, device=device)
            with torch.no_grad():
                fake = model.generator(z, label)
            real_scores, real_cls = model.discriminator(signal, label)
            fake_scores, fake_cls = model.discriminator(fake,   label)
            gp = gradient_penalty(model.discriminator, signal, fake, label, device)
            # WGAN-GP loss + auxiliary classification on real samples
            d_loss = discriminator_loss(real_scores, fake_scores) + GP_LAMBDA * gp + CLS_LAMBDA * nn.CrossEntropyLoss()(real_cls, label)
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

        # train generator: WGAN loss + auxiliary classification
        z = torch.randn(B, model.latent_dim, device=device)
        fake = model.generator(z, label)
        fake_scores, fake_cls = model.discriminator(fake, label)
        g_loss = generator_loss(fake_scores) + CLS_LAMBDA * nn.CrossEntropyLoss()(fake_cls, label)
        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()

        d_loss_sum += d_loss.item()
        g_loss_sum += g_loss.item()
        n += 1

    return d_loss_sum / n, g_loss_sum / n

# mlp training
T_MAX = 70.0
N_TIMESTEPS = 36
def fit_sample_mlp(signal: torch.Tensor, label: int, epochs: int = 2000, lr: float = 1e-3, device: torch.device = torch.device('cpu')) -> tuple[BreathMLP, float]:
    model = BreathMLP().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    t_data = torch.linspace(0, T_MAX, N_TIMESTEPS, device=device) / T_MAX
    y_obs  = signal.to(device).T

    for _ in range(epochs):
        optimiser.zero_grad()
        loss = nn.functional.mse_loss(model(t_data, label), y_obs)
        loss.backward()
        optimiser.step()

    with torch.no_grad():
        mse = nn.functional.mse_loss(model(t_data, label), y_obs).item()

    return model.cpu(), mse

# main loop
def main():
    args = parse_args()

    # paths
    os.makedirs(f'results/{args.model}', exist_ok=True)
    if args.model == 'vae':
        run_id = f'{args.region}_s{args.seed}_ld{args.latent_dim}_fb{args.free_bits}'
    elif args.model == 'cvae':
        run_id = f'{args.region}_s{args.seed}_ld{args.latent_dim}_ed{args.embed_dim}_fb{args.free_bits}'
    elif args.model == 'cvae_part':
        run_id = f'{args.region}_s{args.seed}_ld{args.latent_dim}_ed{args.embed_dim}_pd{args.part_embed_dim}_fb{args.free_bits}'
    elif args.model == 'mlp':
        run_id = f'{args.region}_s{args.seed}'
    else:
        run_id = f'{args.region}_s{args.seed}'
    ckpt_path = f'results/{args.model}/{run_id}_checkpoint.pt'
    history_path = f'results/{args.model}/{run_id}_train_history.csv'

    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Model: {args.model} | Region: {args.region} | Device: {device} | Epochs: {args.epochs}')

    # data set
    df = load_dataset(args.dataset_dir)
    df = df[df['region'] == args.region].reset_index(drop=True)
    df_train, df_val, _ = split_dataset(df, random_state=args.seed)
    train_ds = BreathDataset(df_train)
    val_ds = BreathDataset(df_val, stats=train_ds.stats)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    print(f'[TRAIN] Trials for {args.region}: {len(df)}')

    # vae, cvae, and cvae_part branch
    if args.model in ('vae', 'cvae', 'cvae_part'):
        use_participant = args.model == 'cvae_part'
        conditional = args.model in ('cvae', 'cvae_part')
        if args.model == 'cvae_part':
            model = CVAE(latent_dim=args.latent_dim, embed_dim=args.embed_dim, condition_on_participant=True, part_embed_dim=args.part_embed_dim).to(device)
        elif args.model == 'cvae':
            model = CVAE(latent_dim=args.latent_dim, embed_dim=args.embed_dim).to(device)
        else:
            model = VAE(latent_dim=args.latent_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_val_loss = torch.inf
        history = []

        for epoch in range(1, args.epochs + 1):
            train_loss, train_recon, train_kl = train_vae_one_epoch(model, train_loader, optimizer, epoch, args.epochs, device, args.free_bits, conditional, use_participant)
            val_loss, val_recon, val_kl = evaluate(model, val_loader, epoch, args.epochs, device, args.free_bits, conditional, use_participant)
            scheduler.step()

            if val_loss < best_val_loss and beta_schedule(epoch, args.epochs) >= 1.0:
                best_val_loss = val_loss
                ckpt = {'epoch': epoch, 'model_state': model.state_dict(),
                        'stats': train_ds.stats, 'latent_dim': args.latent_dim,
                        'region': args.region}
                if args.model in ('cvae', 'cvae_part'):
                    ckpt['embed_dim'] = args.embed_dim
                if args.model == 'cvae_part':
                    ckpt['part_embed_dim'] = args.part_embed_dim
                torch.save(ckpt, ckpt_path)

            beta = beta_schedule(epoch, args.epochs)
            n_active = active_dims(model, train_ds, device, conditional=conditional, use_participant=use_participant) if epoch % args.log_every == 0 or epoch == 1 else history[-1]['active_dims'] if history else 0
            history.append({'epoch': epoch, 'beta': beta,
                            'train_loss': train_loss, 'train_recon': train_recon, 'train_kl': train_kl,
                            'val_loss': val_loss, 'val_recon': val_recon, 'val_kl': val_kl,
                            'active_dims': n_active})
            if epoch % args.log_every == 0 or epoch == 1:
                print(f'[TRAIN] Epoch {epoch:4d}/{args.epochs} | beta={beta:.2f} | '
                      f'train loss={train_loss:.4f} (recon={train_recon:.4f}, kl={train_kl:.4f}) | '
                      f'val loss={val_loss:.4f} (recon={val_recon:.4f}, kl={val_kl:.4f}) | '
                      f'active_dims={n_active}')

        print(f'[TRAIN] Best val loss: {best_val_loss} | Checkpoint: {ckpt_path}')

    # gan branch
    elif args.model == 'gan':
        model = CGAN(latent_dim=args.latent_dim, embed_dim=args.embed_dim).to(device)
        lr_g = args.lr_g if args.lr_g is not None else args.lr
        lr_d = args.lr_d if args.lr_d is not None else args.lr
        opt_g = torch.optim.Adam(model.generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))

        best_g_loss = torch.inf
        history = []

        for epoch in range(1, args.epochs + 1):
            d_loss, g_loss = train_gan_one_epoch(model, train_loader, opt_g, opt_d, device, args.n_critic)
            # save checkpoint when loss improves
            if g_loss < best_g_loss:
                best_g_loss = g_loss
                torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                            'stats': train_ds.stats, 'latent_dim': args.latent_dim,
                            'embed_dim': args.embed_dim, 'region': args.region}, ckpt_path)

            history.append({'epoch': epoch, 'd_loss': d_loss, 'g_loss': g_loss})
            if epoch % args.log_every == 0 or epoch == 1:
                print(f'[TRAIN] Epoch {epoch:4d}/{args.epochs} | D loss={d_loss:.4f} | G loss={g_loss:.4f}')

        print(f'[TRAIN] Best G loss: {best_g_loss} | Checkpoint: {ckpt_path}')

    # mlp branch
    elif args.model == 'mlp':
        out_dir = f'results/mlp/{run_id}'
        os.makedirs(out_dir, exist_ok=True)
        fitted: list[tuple[BreathMLP, int, int]] = []
        print(f'[TRAIN] Fitting {len(train_ds)} samples, {args.epochs} epochs each:')
        for i in tqdm(range(len(train_ds)), total=len(train_ds), desc='fitting'):
            signal, _, label, _ = train_ds[i]
            model, _ = fit_sample_mlp(signal, label, epochs=args.epochs, device=device)
            fitted.append((model, label, i))
            torch.save(model.state_dict(), os.path.join(out_dir, f'sample_{i}.pt'))
        return

    # save training history (vae/cvae/gan)
    with open(history_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(f'[TRAIN] History saved to {history_path}')

if __name__ == '__main__':
    main()
