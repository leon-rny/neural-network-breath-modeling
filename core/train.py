import argparse
import csv
import os

import torch
from torch.utils.data import DataLoader

from core.data import BreathDataset, load_dataset, split_dataset
from models.vae import VAE, CVAE, elbo_loss
from models.gan import CGAN, discriminator_loss, generator_loss, gradient_penalty

# cli arguments
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train VAE/CVAE/GAN on breath signals')
    p.add_argument('--model', required=True, choices=['vae', 'cvae', 'gan'])
    p.add_argument('--region', required=True, choices=['mouth', 'nose'])
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--lr_g', type=float, default=None)
    p.add_argument('--lr_d', type=float, default=None)
    p.add_argument('--n_critic', type=int, default=5)
    p.add_argument('--latent_dim', type=int, default=32)
    p.add_argument('--embed_dim', type=int, default=16)
    p.add_argument('--dataset_dir', default='dataset')
    p.add_argument('--log_every', type=int, default=10)
    return p.parse_args()

# vae and cvae training
def beta_schedule(epoch: int, total_epochs: int) -> float:
    """Linear beta anneal: 0 -> 1 over first half of training to avoid posterior collapse."""
    return min(1.0, epoch / (total_epochs * 0.5))

def train_one_epoch(model, loader, optimizer, epoch, total_epochs, device, conditional=False):
    model.train()
    beta = beta_schedule(epoch, total_epochs)
    total_loss = recon_sum = kl_sum = 0.0
    for signal, _time, label in loader:
        signal = signal.to(device)
        if conditional:
            label = label.long().to(device)
            x_hat, mu, logvar = model(signal, label)
        else:
            x_hat, mu, logvar = model(signal)
        loss, recon, kl = elbo_loss(signal, x_hat, mu, logvar, beta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        recon_sum  += recon.item()
        kl_sum     += kl.item()
    n = len(loader)
    return total_loss / n, recon_sum / n, kl_sum / n

@torch.no_grad()
def evaluate(model, loader, epoch, total_epochs, device, conditional=False):
    model.eval()
    beta = beta_schedule(epoch, total_epochs)
    total_loss = recon_sum = kl_sum = 0.0
    for signal, _time, label in loader:
        signal = signal.to(device)
        if conditional:
            label = label.long().to(device)
            x_hat, mu, logvar = model(signal, label)
        else:
            x_hat, mu, logvar = model(signal)
        loss, recon, kl = elbo_loss(signal, x_hat, mu, logvar, beta)
        total_loss += loss.item()
        recon_sum  += recon.item()
        kl_sum     += kl.item()
    n = len(loader)
    return total_loss / n, recon_sum / n, kl_sum / n

# gan training
GP_LAMBDA = 10 # gradient penalty coefficient
def train_gan_one_epoch(model, loader, opt_g, opt_d, device, n_critic):
    model.train()
    d_loss_sum = g_loss_sum = 0.0
    n = 0
    for signal, _time, label in loader:
        signal = signal.to(device)
        label = label.long().to(device)
        B = signal.size(0)

        # train critic n_critic times per generator update
        for _ in range(n_critic):
            z = torch.randn(B, model.latent_dim, device=device)
            with torch.no_grad():
                fake = model.generator(z, label)
            real_scores = model.discriminator(signal, label)
            fake_scores = model.discriminator(fake,   label)
            gp     = gradient_penalty(model.discriminator, signal, fake, label, device)
            d_loss = discriminator_loss(real_scores, fake_scores) + GP_LAMBDA * gp
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

        # train generator
        z = torch.randn(B, model.latent_dim, device=device)
        fake = model.generator(z, label)
        fake_scores = model.discriminator(fake, label)
        g_loss = generator_loss(fake_scores)
        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()

        d_loss_sum += d_loss.item()
        g_loss_sum += g_loss.item()
        n += 1

    return d_loss_sum / n, g_loss_sum / n

# main loop
def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps'  if torch.backends.mps.is_available() else 'cpu')
    print(f'Model: {args.model} | Region: {args.region} | Device: {device} | Epochs: {args.epochs}')

    # data
    df = load_dataset(args.dataset_dir)
    df = df[df['region'] == args.region].reset_index(drop=True)
    print(f'Trials for {args.region}: {len(df)}')

    df_train, df_val, df_test = split_dataset(df, random_state=42)
    train_ds = BreathDataset(df_train)
    val_ds   = BreathDataset(df_val, stats=train_ds.stats)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    os.makedirs('models/checkpoints', exist_ok=True)
    os.makedirs(f'results/{args.model}', exist_ok=True)
    ckpt_path    = f'models/checkpoints/{args.model}_{args.region}.pt'
    history_path = f'results/{args.model}/train_{args.region}.csv'

    # vae and cvae branch
    if args.model in ('vae', 'cvae'):
        conditional = args.model == 'cvae'
        model = (CVAE(latent_dim=args.latent_dim, embed_dim=args.embed_dim)
                 if conditional else VAE(latent_dim=args.latent_dim)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_val_loss = torch.inf
        history = []

        for epoch in range(1, args.epochs + 1):
            train_loss, train_recon, train_kl = train_one_epoch(
                model, train_loader, optimizer, epoch, args.epochs, device, conditional)
            val_loss, val_recon, val_kl = evaluate(
                model, val_loader, epoch, args.epochs, device, conditional)
            scheduler.step()

            if val_loss < best_val_loss and beta_schedule(epoch, args.epochs) >= 1.0:
                best_val_loss = val_loss
                ckpt = {'epoch': epoch, 'model_state': model.state_dict(),
                        'stats': train_ds.stats, 'latent_dim': args.latent_dim,
                        'region': args.region}
                if conditional:
                    ckpt['embed_dim'] = args.embed_dim
                torch.save(ckpt, ckpt_path)

            beta = beta_schedule(epoch, args.epochs)
            history.append({'epoch': epoch, 'beta': beta,
                            'train_loss': train_loss, 'train_recon': train_recon, 'train_kl': train_kl,
                            'val_loss':   val_loss,   'val_recon':   val_recon,   'val_kl':   val_kl})
            if epoch % args.log_every == 0 or epoch == 1:
                print(f'Epoch {epoch:4d}/{args.epochs} | beta={beta:.2f} | '
                      f'train loss={train_loss:.4f} (recon={train_recon:.4f}, kl={train_kl:.4f}) | '
                      f'val loss={val_loss:.4f} (recon={val_recon:.4f}, kl={val_kl:.4f})')

        print(f'Best val loss: {best_val_loss:.4f} | Checkpoint: {ckpt_path}')

    # gan branch
    elif args.model == 'gan':
        model  = CGAN(latent_dim=args.latent_dim, embed_dim=args.embed_dim).to(device)
        lr_g   = args.lr_g if args.lr_g is not None else args.lr
        lr_d   = args.lr_d if args.lr_d is not None else args.lr
        opt_g  = torch.optim.Adam(model.generator.parameters(),     lr=lr_g, betas=(0.5, 0.999))
        opt_d  = torch.optim.Adam(model.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))

        best_g_loss = torch.inf
        history = []

        for epoch in range(1, args.epochs + 1):
            d_loss, g_loss = train_gan_one_epoch(
                model, train_loader, opt_g, opt_d, device, args.n_critic)
            # save checkpoint when loss improves
            if g_loss < best_g_loss:
                best_g_loss = g_loss
                torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                            'stats': train_ds.stats, 'latent_dim': args.latent_dim,
                            'embed_dim': args.embed_dim, 'region': args.region}, ckpt_path)

            history.append({'epoch': epoch, 'd_loss': d_loss, 'g_loss': g_loss})
            if epoch % args.log_every == 0 or epoch == 1:
                print(f'Epoch {epoch:4d}/{args.epochs} | D loss={d_loss:.4f} | G loss={g_loss:.4f}')

        print(f'Best G loss: {best_g_loss:.4f} | Checkpoint: {ckpt_path}')

    # save training complete history
    with open(history_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(f'Training history saved to {history_path}')

if __name__ == '__main__':
    main()
