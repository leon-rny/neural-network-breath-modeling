import argparse
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.data import BreathDataset, PhysicsInformedDataset, load_dataset, get_split
from core.utils import seed_everything, seed_worker, make_generator
from models.vae import VAE, CVAE, elbo_loss
from models.pinn import PhysicsInformedCVAE

def _cv_marker(cv_mode: str) -> str:
    return '' if cv_mode == 'kfold' else f'_{cv_mode}'

def _drop_marker(part_dropout: float) -> str:
    return '' if part_dropout == 0.0 else f'_drop{part_dropout}'

# cli arguments
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # general
    p.add_argument('--model', required=True, choices=['vae', 'cvae', 'cvae_part', 'pinn'])
    p.add_argument('--region', required=True, choices=['mouth', 'nose'])
    p.add_argument('--dataset_dir', default='dataset')
    p.add_argument('--log_every', type=int, default=25)
    p.add_argument('--init_seed', type=int, default=42, help='Model-internal randomness; varies across runs to characterize sensitivity.')
    p.add_argument('--split_seed', type=int, default=42, help='Data partition; fixed for paired comparisons.')
    p.add_argument('--fold', type=int, default=1, help='1-indexed fold in [1, n_folds] (kfold) or [1, n_subjects] (loso).')
    p.add_argument('--n_folds', type=int, default=5)
    p.add_argument('--cv_mode', choices=['kfold', 'loso'], default='kfold', help='kfold: split by trial; loso: leave-one-subject-out (fold count derived from data).')
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--batch_size', type=int, default=32)
    # model-specific
    p.add_argument('--latent_dim', type=int, default=16)   # ablation default
    p.add_argument('--embed_dim', type=int, default=8)     # ablation default
    p.add_argument('--part_embed_dim', type=int, default=8)
    p.add_argument('--free_bits', type=float, default=0.0)
    p.add_argument('--part_dropout', type=float, default=0.0, help='CFG-style participant-dropout prob (cvae_part/pinn). Required >0 for LOSO null-token generation.')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--beta_max', type=float, default=0.1)
    p.add_argument('--beta_warmup_epochs', type=int, default=None, help='Epochs to linearly warm beta 0->beta_max. Defaults to epochs // 2 (matches the ablations).')
    # pinn-specific (ignored by vae/cvae/cvae_part)
    p.add_argument('--lambda_phys', type=float, default=0.0, help='Weight of the physics-consistency penalty (pinn only).')
    p.add_argument('--tau_s', type=float, default=15.0, help='Sensor time constant for the CIR convolution (pinn only).')
    p.add_argument('--learn_cir_params', action='store_true', help='Learn per-sample CIR params from z (pinn only).')
    # jittering augmentation (training only; defaults = off)
    p.add_argument('--alpha', type=float, default=0.0)
    p.add_argument('--n_copies', type=int, default=1)
    return p.parse_args()

# vae and cvae training
def beta_capped(epoch: int, warmup_epochs: int, beta_max: float = 0.1) -> float:
    """Linear warmup from 0 → beta_max over first half of training, then hold."""
    return min(beta_max, (epoch / warmup_epochs) * beta_max)

def train_vae_one_epoch(model, loader, optimizer, epoch, warmup_epochs, device, free_bits, conditional=False, use_participant=False, beta_max=0.1):
    model.train()
    beta = beta_capped(epoch, warmup_epochs, beta_max=beta_max)
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
def evaluate(model, loader, epoch, warmup_epochs, device, free_bits, conditional=False, use_participant=False, beta_max=0.1):
    model.eval()
    beta = beta_capped(epoch, warmup_epochs, beta_max=beta_max)
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
        recon_sum += recon.item()
        kl_sum += kl.item()
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

# pinn training
PINN_DIAG_KEYS = ('u_norm', 'logA_mean', 'logA_std', 'logD_mean', 'logD_std', 'logv_mean', 'logv_std')

def pinn_diag_stats(u_post_softplus, cir_params):
    log_A, log_D, log_v = cir_params
    return {'u_norm': u_post_softplus.norm(dim=1).mean().item(),
            'logA_mean': log_A.mean().item(), 'logA_std': log_A.std().item(),
            'logD_mean': log_D.mean().item(), 'logD_std': log_D.std().item(),
            'logv_mean': log_v.mean().item(), 'logv_std': log_v.std().item()}

def train_pinn_one_epoch(model, loader, optimizer, epoch, warmup_epochs, device, free_bits, lambda_phys, beta_max=0.1):
    model.train()
    beta = beta_capped(epoch, warmup_epochs, beta_max=beta_max)
    totals = {k: 0.0 for k in ('total', 'recon', 'kl', 'phys') + PINN_DIAG_KEYS}
    for signal, _time, label, participant, _onset in loader:
        signal = signal.to(device)
        label = label.long().to(device)
        participant = participant.long().to(device)
        x_hat, mu, logvar, u_post_softplus, cir_params, humidity_phys = model(signal, label, participant)
        elbo, recon, kl = elbo_loss(signal, x_hat, mu, logvar, beta, free_bits)
        phys = F.mse_loss(humidity_phys, x_hat[:, 0, :])
        loss = elbo + lambda_phys * phys
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            diag = pinn_diag_stats(u_post_softplus, cir_params)
        totals['total'] += loss.item()
        totals['recon'] += recon.item()
        totals['kl'] += kl.item()
        totals['phys'] += phys.item()
        for k in PINN_DIAG_KEYS:
            totals[k] += diag[k]
    n = len(loader)
    return {k: v / n for k, v in totals.items()}

@torch.no_grad()
def evaluate_pinn(model, loader, epoch, warmup_epochs, device, free_bits, lambda_phys, beta_max=0.1):
    model.eval()
    beta = beta_capped(epoch, warmup_epochs, beta_max=beta_max)
    totals = {k: 0.0 for k in ('total', 'recon', 'kl', 'phys') + PINN_DIAG_KEYS}
    for signal, _time, label, participant, _onset in loader:
        signal = signal.to(device)
        label = label.long().to(device)
        participant = participant.long().to(device)
        x_hat, mu, logvar, u_post_softplus, cir_params, humidity_phys = model(signal, label, participant)
        elbo, recon, kl = elbo_loss(signal, x_hat, mu, logvar, beta, free_bits)
        phys = F.mse_loss(humidity_phys, x_hat[:, 0, :])
        loss = elbo + lambda_phys * phys
        diag = pinn_diag_stats(u_post_softplus, cir_params)
        totals['total'] += loss.item()
        totals['recon'] += recon.item()
        totals['kl'] += kl.item()
        totals['phys'] += phys.item()
        for k in PINN_DIAG_KEYS:
            totals[k] += diag[k]
    n = len(loader)
    return {k: v / n for k, v in totals.items()}

# main loop
def main():
    args = parse_args()
    if args.beta_warmup_epochs is None:
        args.beta_warmup_epochs = args.epochs // 2
    device = torch.device('cpu')
    print(f'[TRAIN] init_seed={args.init_seed} split_seed={args.split_seed} fold={args.fold} cv={args.cv_mode} | Model: {args.model} | Region: {args.region} | Device: {device} | Epochs: {args.epochs}')

    os.makedirs(f'results/{args.model}', exist_ok=True)
    if args.model == 'vae':
        run_id = f'{args.region}_s{args.init_seed}_ld{args.latent_dim}_fb{args.free_bits}'
    elif args.model == 'cvae':
        run_id = f'{args.region}_s{args.init_seed}_ld{args.latent_dim}_ed{args.embed_dim}_fb{args.free_bits}'
    elif args.model == 'cvae_part':
        run_id = f'{args.region}_s{args.init_seed}_ld{args.latent_dim}_ed{args.embed_dim}_pd{args.part_embed_dim}_fb{args.free_bits}'
    elif args.model == 'pinn':
        run_id = f'{args.region}_s{args.init_seed}_ld{args.latent_dim}_ed{args.embed_dim}_phys{args.lambda_phys}'
    run_id += f'_f{args.fold}{_cv_marker(args.cv_mode)}{_drop_marker(args.part_dropout)}'
    if args.alpha > 0 and args.n_copies > 1:
        run_id += f'_a{args.alpha}_n{args.n_copies}'
    ckpt_path = f'results/{args.model}/{run_id}_checkpoint.pt'
    history_path = f'results/{args.model}/{run_id}_train_history.csv'

    # reproducibility
    seed_everything(args.init_seed)
    g = make_generator(args.init_seed)

    # dataset
    df = load_dataset(args.dataset_dir)
    df = df[df['region'] == args.region].reset_index(drop=True)
    num_participants = df['participant'].nunique()  # embedding table covers all subjects; derive before the split
    # args.fold is 1-indexed
    df_train, df_val, _ = get_split(df, cv_mode=args.cv_mode, fold=args.fold - 1, n_folds=args.n_folds, split_seed=args.split_seed)
    DatasetCls = PhysicsInformedDataset if args.model == 'pinn' else BreathDataset
    train_ds = DatasetCls(df_train, alpha=args.alpha, n_copies=args.n_copies)
    train_ds_clean = DatasetCls(df_train, stats=train_ds.stats) if args.alpha > 0 else train_ds
    val_ds = DatasetCls(df_val, stats=train_ds.stats)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    if args.alpha > 0 and args.n_copies > 1:
        print(f'[TRAIN] Jitter: alpha={args.alpha}, n_copies={args.n_copies} ({len(train_ds_clean)} → {len(train_ds)} samples)')

    # vae, cvae, and cvae_part branch
    if args.model in ('vae', 'cvae', 'cvae_part'):
        use_participant = args.model == 'cvae_part'
        conditional = args.model in ('cvae', 'cvae_part')
        if args.model == 'cvae_part':
            model = CVAE(latent_dim=args.latent_dim, embed_dim=args.embed_dim, condition_on_participant=True, num_participants=num_participants, part_embed_dim=args.part_embed_dim, part_dropout=args.part_dropout).to(device)
        elif args.model == 'cvae':
            model = CVAE(latent_dim=args.latent_dim, embed_dim=args.embed_dim).to(device)
        else:
            model = VAE(latent_dim=args.latent_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_val_loss = torch.inf
        history = []

        epoch_bar = tqdm(range(1, args.epochs + 1), desc=f'[TRAIN] {args.model}', unit='epoch')
        for epoch in epoch_bar:
            train_loss, train_recon, train_kl = train_vae_one_epoch(model, train_loader, optimizer, epoch, args.beta_warmup_epochs, device, args.free_bits, conditional, use_participant, beta_max=args.beta_max)
            val_loss, val_recon, val_kl = evaluate(model, val_loader, epoch, args.beta_warmup_epochs, device, args.free_bits, conditional, use_participant, beta_max=args.beta_max)
            scheduler.step()

            if val_loss < best_val_loss and beta_capped(epoch, args.beta_warmup_epochs, beta_max=args.beta_max) >= args.beta_max:
                best_val_loss = val_loss
                ckpt = {'epoch': epoch, 'model_state': model.state_dict(),
                        'stats': train_ds.stats, 'latent_dim': args.latent_dim,
                        'region': args.region, 'cv_mode': args.cv_mode,
                        'num_participants': num_participants, 'part_dropout': args.part_dropout}
                if args.model in ('cvae', 'cvae_part'):
                    ckpt['embed_dim'] = args.embed_dim
                if args.model == 'cvae_part':
                    ckpt['part_embed_dim'] = args.part_embed_dim
                torch.save(ckpt, ckpt_path)

            beta = beta_capped(epoch, args.beta_warmup_epochs, beta_max=args.beta_max)
            n_active = active_dims(model, train_ds_clean, device, conditional=conditional, use_participant=use_participant) if epoch % args.log_every == 0 or epoch == 1 else history[-1]['active_dims'] if history else 0
            history.append({'epoch': epoch, 'beta': beta,
                            'train_loss': train_loss, 'train_recon': train_recon, 'train_kl': train_kl,
                            'val_loss': val_loss, 'val_recon': val_recon, 'val_kl': val_kl,
                            'active_dims': n_active})
            epoch_bar.set_postfix({'beta': f'{beta:.2f}',
                                   'train': f'{train_loss:.4f}',
                                   'val': f'{val_loss:.4f}',
                                   'active': n_active})

    # pinn branch
    elif args.model == 'pinn':
        params_cir = np.load(f'results/pinn/params_{args.region}.npy')
        t_grid = np.arange(36) * 2.0
        model = PhysicsInformedCVAE(cir_params_init=params_cir, t_grid=t_grid, tau_s=args.tau_s,
                                    learn_cir_params=args.learn_cir_params, latent_dim=args.latent_dim,
                                    num_classes=3, embed_dim=args.embed_dim, condition_on_participant=True,
                                    num_participants=num_participants, part_embed_dim=args.part_embed_dim,
                                    part_dropout=args.part_dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_val_loss = torch.inf
        history = []

        epoch_bar = tqdm(range(1, args.epochs + 1), desc=f'[TRAIN] {args.model}', unit='epoch')
        for epoch in epoch_bar:
            train_m = train_pinn_one_epoch(model, train_loader, optimizer, epoch, args.beta_warmup_epochs, device, args.free_bits, args.lambda_phys, beta_max=args.beta_max)
            val_m = evaluate_pinn(model, val_loader, epoch, args.beta_warmup_epochs, device, args.free_bits, args.lambda_phys, beta_max=args.beta_max)
            scheduler.step()

            beta = beta_capped(epoch, args.beta_warmup_epochs, beta_max=args.beta_max)
            # select on the ELBO (recon + beta*KL), excluding the physics penalty — comparable across the lambda_phys sweep and to the CVAE
            val_elbo = val_m['recon'] + beta * val_m['kl']
            if val_elbo < best_val_loss and beta >= args.beta_max:
                best_val_loss = val_elbo
                torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                            'stats': train_ds.stats, 'latent_dim': args.latent_dim,
                            'embed_dim': args.embed_dim, 'part_embed_dim': args.part_embed_dim,
                            'region': args.region, 'cir_params': params_cir, 't_grid': t_grid,
                            'tau_s': args.tau_s, 'learn_cir_params': args.learn_cir_params,
                            'condition_on_participant': True, 'num_participants': num_participants,
                            'part_dropout': args.part_dropout, 'cv_mode': args.cv_mode}, ckpt_path)

            history.append({'epoch': epoch, 'beta': beta,
                            **{f'train_{k}': train_m[k] for k in ('total', 'recon', 'kl', 'phys')},
                            **{f'val_{k}': val_m[k] for k in ('total', 'recon', 'kl', 'phys')},
                            **{k: train_m[k] for k in PINN_DIAG_KEYS}})
            epoch_bar.set_postfix({'beta': f'{beta:.2f}',
                                   'tr': f'{train_m["recon"]:.4f}',
                                   'vr': f'{val_m["recon"]:.4f}',
                                   'kl': f'{val_m["kl"]:.4f}',
                                   'ph': f'{val_m["phys"]:.4f}'})

    # save training history
    with open(history_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(f'[TRAIN] History saved to {history_path}\n[TRAIN] Checkpoint saved to {ckpt_path}')

if __name__ == '__main__':
    main()
