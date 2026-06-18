import argparse
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.data import BreathDataset, PhysicsInformedDataset, load_dataset, get_split, subset_tag, cir_marker, phys_prep_marker
from core.utils import seed_everything, seed_worker, make_generator
from models.vae import VAE, CVAE, elbo_loss
from models.pinn import PhysicsInformedCVAE, SharedTransportPINN
from models.diffusion import ConditionalDiffusion
from models.gan import Generator, Discriminator

def _cv_marker(cv_mode: str) -> str:
    """Run-id suffix for the cross-validation mode ('' for kfold)."""
    return '' if cv_mode == 'kfold' else f'_{cv_mode}'

def _drop_marker(part_dropout: float) -> str:
    """Run-id suffix for the participant-dropout rate ('' when off)."""
    return '' if part_dropout == 0.0 else f'_drop{part_dropout}'

# cli arguments
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser()
    # general
    p.add_argument('--model', required=True, choices=['vae', 'cvae', 'cvae_part', 'pinn', 'tpinn', 'diffusion', 'gan'])
    p.add_argument('--n_steps', type=int, default=200)
    p.add_argument('--region', required=True, choices=['mouth', 'nose'])
    p.add_argument('--dataset_dir', default='dataset')
    p.add_argument('--log_every', type=int, default=25)
    p.add_argument('--init_seed', type=int, default=42)
    p.add_argument('--split_seed', type=int, default=42)
    p.add_argument('--fold', type=int, default=1)
    p.add_argument('--n_folds', type=int, default=5)
    p.add_argument('--cv_mode', choices=['kfold', 'loso'], default='kfold')
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--batch_size', type=int, default=32)
    # model-specific
    p.add_argument('--latent_dim', type=int, default=16)   # ablation default
    p.add_argument('--embed_dim', type=int, default=8)     # ablation default
    p.add_argument('--part_embed_dim', type=int, default=8)
    p.add_argument('--free_bits', type=float, default=0.0)
    p.add_argument('--part_dropout', type=float, default=0.0)
    p.add_argument('--include_subjects', default='')
    p.add_argument('--cir_tag', default='')
    p.add_argument('--phys_prep', choices=['peakscale', 'shared', 'stdscale'], default='peakscale')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--beta_max', type=float, default=0.1)
    p.add_argument('--beta_warmup_epochs', type=int, default=None)
    # pinn-specific (ignored by vae/cvae/cvae_part)
    p.add_argument('--lambda_phys', type=float, default=0.0)
    p.add_argument('--tau_s', type=float, default=15.0)
    p.add_argument('--learn_cir_params', action='store_true')
    p.add_argument('--phys_residual', action='store_true')
    p.add_argument('--class_transport', action='store_true')
    p.add_argument('--parametric_source', action='store_true')
    # jittering augmentation (training only; defaults = off)
    p.add_argument('--alpha', type=float, default=0.0)
    p.add_argument('--n_copies', type=int, default=1)
    p.add_argument('--hp_tag', default='')
    p.add_argument('--subj_adv_lambda', type=float, default=0.0)
    p.add_argument('--diff_hidden', type=int, default=64)
    p.add_argument('--gan_hidden', type=int, default=64)
    p.add_argument('--gan_lr_d', type=float, default=None)
    p.add_argument('--gan_loss', choices=['bce', 'hinge'], default='bce')
    return p.parse_args()

# vae and cvae training
def beta_capped(epoch: int, warmup_epochs: int, beta_max: float = 0.1) -> float:
    """Linear warmup from 0 -> beta_max over first half of training, then hold."""
    return min(beta_max, (epoch / warmup_epochs) * beta_max)

def train_vae_one_epoch(model, loader, optimizer, epoch, warmup_epochs, device, free_bits, conditional=False, use_participant=False, beta_max=0.1):
    """Train a (C)VAE for one epoch with beta-warmup ELBO.

    :param conditional: pass the class label to the model (cvae/cvae_part).
    :param use_participant: also pass the participant index (cvae_part).
    :return: epoch-mean (loss, recon, kl).
    """
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

def train_cvae_adv_one_epoch(model, loader, optimizer, epoch, warmup_epochs, device, free_bits, adv_lambda, beta_max=0.1):
    """cvae_part + subject-adversarial: loss = ELBO + CE(participant predicted from z through a GRL),
    so the encoder is pushed to drop subject identity (subject-invariant latent for LOSO). Replicates
    the null-token participant dropout the standard forward applies. Returns (loss, recon, kl)."""
    model.train()
    beta = beta_capped(epoch, warmup_epochs, beta_max=beta_max)
    total = recon_sum = kl_sum = 0.0
    for signal, _time, label, participant in loader:
        signal = signal.to(device)
        label = label.long().to(device)
        participant = participant.long().to(device)
        p_in = participant
        if model._cond_part and model.part_dropout > 0.0 and torch.rand(1).item() < model.part_dropout:
            p_in = torch.full_like(participant, model.null_part_idx)
        mu, logvar = model.encoder(signal, label, p_in)
        z = model.reparameterize(mu, logvar)
        x_hat = model.decoder(z, label, p_in)
        elbo, recon, kl = elbo_loss(signal, x_hat, mu, logvar, beta, free_bits)
        adv = F.cross_entropy(model.adv_logits(z, adv_lambda), participant)  # GRL flips sign to the encoder
        loss = elbo + adv
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
        recon_sum += recon.item()
        kl_sum += kl.item()
    n = len(loader)
    return total / n, recon_sum / n, kl_sum / n

@torch.no_grad()
def evaluate(model, loader, epoch, warmup_epochs, device, free_bits, conditional=False, use_participant=False, beta_max=0.1):
    """Evaluate a (C)VAE for one epoch (no grad).

    :param conditional: pass the class label to the model (cvae/cvae_part).
    :param use_participant: also pass the participant index (cvae_part).
    :return: epoch-mean (loss, recon, kl).
    """
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
    """Count latent dims whose mean per-dim KL exceeds threshold (no grad).

    :return: number of active latent dimensions.
    """
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
    """Per-batch diagnostics for PINN training.

    :param u_post_softplus: (B, 36) physics source after softplus.
    :param cir_params: (log_A, log_D, log_v) transport-parameter tensors.
    :return: dict keyed by PINN_DIAG_KEYS (source norm, per-param mean/std).
    """
    log_A, log_D, log_v = cir_params
    return {'u_norm': u_post_softplus.norm(dim=1).mean().item(),
            'logA_mean': log_A.mean().item(), 'logA_std': log_A.std().item(),
            'logD_mean': log_D.mean().item(), 'logD_std': log_D.std().item(),
            'logv_mean': log_v.mean().item(), 'logv_std': log_v.std().item()}

def train_pinn_one_epoch(model, loader, optimizer, epoch, warmup_epochs, device, free_bits, lambda_phys, beta_max=0.1):
    """Train a (t)PINN for one epoch: loss = ELBO + lambda_phys * MSE(humidity_phys, x_hat humidity).

    :return: dict of epoch-mean metrics ('total', 'recon', 'kl', 'phys') plus PINN_DIAG_KEYS.
    """
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
    """Evaluate a (t)PINN for one epoch (no grad), same loss as training.

    :return: dict of epoch-mean metrics ('total', 'recon', 'kl', 'phys') plus PINN_DIAG_KEYS.
    """
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
    """Entry point: parse args, build the dataset split, train the requested model, save checkpoint and history."""
    args = parse_args()
    if args.beta_warmup_epochs is None:
        args.beta_warmup_epochs = args.epochs // 2
    device = torch.device('cpu')
    print(f'[TRAIN] init_seed={args.init_seed} split_seed={args.split_seed} fold={args.fold} cv={args.cv_mode} | Model: {args.model} | Region: {args.region} | Device: {device} | Epochs: {args.epochs}')

    include_subjects = tuple(x.strip() for x in args.include_subjects.split(',') if x.strip())
    os.makedirs(f'results/{args.model}', exist_ok=True)
    if args.model == 'vae':
        run_id = f'{args.region}_s{args.init_seed}_ld{args.latent_dim}_fb{args.free_bits}'
    elif args.model == 'cvae':
        run_id = f'{args.region}_s{args.init_seed}_ld{args.latent_dim}_ed{args.embed_dim}_fb{args.free_bits}'
    elif args.model == 'cvae_part':
        run_id = f'{args.region}_s{args.init_seed}_ld{args.latent_dim}_ed{args.embed_dim}_pd{args.part_embed_dim}_fb{args.free_bits}' + (f'_adv{args.subj_adv_lambda}' if args.subj_adv_lambda > 0 else '')
    elif args.model == 'pinn':
        run_id = f'{args.region}_s{args.init_seed}_ld{args.latent_dim}_ed{args.embed_dim}_phys{args.lambda_phys}'
    elif args.model == 'tpinn':
        run_id = f'{args.region}_s{args.init_seed}_ld{args.latent_dim}_ed{args.embed_dim}_tphys' + ('_ct' if args.class_transport else '') + ('_ps' if args.parametric_source else '') + ('_res' if args.phys_residual else '') + ('_learn' if args.learn_cir_params else '')
    elif args.model == 'diffusion':
        run_id = f'{args.region}_s{args.init_seed}_ed{args.embed_dim}_diff_h{args.diff_hidden}_st{args.n_steps}'
    elif args.model == 'gan':
        run_id = f'{args.region}_s{args.init_seed}_ld{args.latent_dim}_ed{args.embed_dim}_gan_h{args.gan_hidden}_{args.gan_loss}_lrd{args.gan_lr_d if args.gan_lr_d is not None else args.lr}'
    run_id += f'_f{args.fold}{_cv_marker(args.cv_mode)}{_drop_marker(args.part_dropout)}{subset_tag(include_subjects)}{cir_marker(args.cir_tag)}{phys_prep_marker(args.phys_prep)}'
    if args.alpha > 0 and args.n_copies > 1:
        run_id += f'_a{args.alpha}_n{args.n_copies}'
    if args.hp_tag:
        run_id += f'_{args.hp_tag}'
    ckpt_path = f'results/{args.model}/{run_id}_checkpoint.pt'
    history_path = f'results/{args.model}/{run_id}_train_history.csv'

    # reproducibility
    seed_everything(args.init_seed)
    g = make_generator(args.init_seed)

    # dataset
    df = load_dataset(args.dataset_dir)
    df = df[df['region'] == args.region].reset_index(drop=True)
    num_participants = df['participant'].nunique()  # embedding table covers all subjects; derive before the split
    if include_subjects:  # restrict to a participant subset (embedding table stays full-size; indices are global)
        df = df[df['participant'].isin(include_subjects)].reset_index(drop=True)
    # args.fold is 1-indexed
    df_train, df_val, _ = get_split(df, cv_mode=args.cv_mode, fold=args.fold - 1, n_folds=args.n_folds, split_seed=args.split_seed)
    DatasetCls = PhysicsInformedDataset if args.model in ('pinn', 'tpinn') else BreathDataset
    ds_kwargs = {'alpha': args.alpha, 'n_copies': args.n_copies}
    if args.model in ('pinn', 'tpinn'):
        ds_kwargs['phys_prep'] = args.phys_prep      # input-normalization mode (stored in stats -> ckpt)
    train_ds = DatasetCls(df_train, **ds_kwargs)
    train_ds_clean = DatasetCls(df_train, stats=train_ds.stats) if args.alpha > 0 else train_ds
    val_ds = DatasetCls(df_val, stats=train_ds.stats)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    if args.alpha > 0 and args.n_copies > 1:
        print(f'[TRAIN] Jitter: alpha={args.alpha}, n_copies={args.n_copies} ({len(train_ds_clean)} -> {len(train_ds)} samples)')

    # vae, cvae, and cvae_part branch
    if args.model in ('vae', 'cvae', 'cvae_part'):
        use_participant = args.model == 'cvae_part'
        conditional = args.model in ('cvae', 'cvae_part')
        if args.model == 'cvae_part':
            model = CVAE(latent_dim=args.latent_dim, embed_dim=args.embed_dim, condition_on_participant=True, num_participants=num_participants, part_embed_dim=args.part_embed_dim, part_dropout=args.part_dropout, subj_adv=(args.subj_adv_lambda > 0)).to(device)
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
            if use_participant and args.subj_adv_lambda > 0:
                train_loss, train_recon, train_kl = train_cvae_adv_one_epoch(model, train_loader, optimizer, epoch, args.beta_warmup_epochs, device, args.free_bits, args.subj_adv_lambda, beta_max=args.beta_max)
            else:
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
                    ckpt['subj_adv'] = args.subj_adv_lambda > 0
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

    # pinn / tpinn branch (tpinn = shared-transport physics-as-decoder; reuses the same train loop with phys=0)
    elif args.model in ('pinn', 'tpinn'):
        params_cir = np.load(f'results/pinn/params_{args.region}{("_" + args.cir_tag) if args.cir_tag else ""}.npy')
        t_grid = np.arange(36) * 2.0
        if args.model == 'tpinn':
            model = SharedTransportPINN(cir_params_init=params_cir, t_grid=t_grid, tau_s=args.tau_s,
                                        learn_transport=args.learn_cir_params, residual=args.phys_residual,
                                        class_transport=args.class_transport, parametric_source=args.parametric_source,
                                        latent_dim=args.latent_dim,
                                        num_classes=3, embed_dim=args.embed_dim, condition_on_participant=True,
                                        num_participants=num_participants, part_embed_dim=args.part_embed_dim,
                                        part_dropout=args.part_dropout).to(device)
        else:
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
            # select on the ELBO (recon + beta*KL), excluding the physics penalty - comparable across the lambda_phys sweep and to the CVAE
            val_elbo = val_m['recon'] + beta * val_m['kl']
            if val_elbo < best_val_loss and beta >= args.beta_max:
                best_val_loss = val_elbo
                torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                            'stats': train_ds.stats, 'latent_dim': args.latent_dim,
                            'embed_dim': args.embed_dim, 'part_embed_dim': args.part_embed_dim,
                            'region': args.region, 'cir_params': params_cir, 't_grid': t_grid,
                            'tau_s': args.tau_s, 'learn_cir_params': args.learn_cir_params,
                            'phys_residual': args.phys_residual, 'class_transport': args.class_transport,
                            'parametric_source': args.parametric_source,
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

    # diffusion branch (DDPM eps-prediction; BreathDataset z-score; save last epoch - stable training)
    elif args.model == 'diffusion':
        model = ConditionalDiffusion(num_classes=3, num_participants=num_participants,
                                     embed_dim=args.embed_dim, part_embed_dim=args.part_embed_dim,
                                     condition_on_participant=True, part_dropout=args.part_dropout,
                                     n_steps=args.n_steps, hidden=args.diff_hidden).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        history = []
        epoch_bar = tqdm(range(1, args.epochs + 1), desc=f'[TRAIN] {args.model}', unit='epoch')
        for epoch in epoch_bar:
            model.train()
            tr = 0.0
            for signal, _time, label, participant in train_loader:
                signal = signal.to(device)
                label = label.long().to(device)
                participant = participant.long().to(device)
                loss = model.loss(signal, label, participant)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tr += loss.item()
            scheduler.step()
            tr /= len(train_loader)
            model.eval()
            vl = 0.0
            with torch.no_grad():
                for signal, _time, label, participant in val_loader:
                    signal = signal.to(device)
                    label = label.long().to(device)
                    participant = participant.long().to(device)
                    vl += model.loss(signal, label, participant).item()
            vl /= max(len(val_loader), 1)
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'stats': train_ds.stats,
                        'embed_dim': args.embed_dim, 'part_embed_dim': args.part_embed_dim, 'n_steps': args.n_steps,
                        'hidden': args.diff_hidden,
                        'region': args.region, 'condition_on_participant': True, 'num_participants': num_participants,
                        'part_dropout': args.part_dropout, 'cv_mode': args.cv_mode}, ckpt_path)
            history.append({'epoch': epoch, 'train_loss': tr, 'val_loss': vl})
            epoch_bar.set_postfix({'tr': f'{tr:.4f}', 'vl': f'{vl:.4f}'})

    # gan branch (conditional 1D GAN; checkpoint = the generator). Last epoch saved (GAN val is ill-defined).
    elif args.model == 'gan':
        z_dim = args.latent_dim
        G = Generator(z_dim=z_dim, num_classes=3, num_participants=num_participants, embed_dim=args.embed_dim,
                      part_embed_dim=args.part_embed_dim, condition_on_participant=True, part_dropout=args.part_dropout,
                      hidden=args.gan_hidden).to(device)
        D = Discriminator(num_classes=3, embed_dim=args.embed_dim, hidden=args.gan_hidden).to(device)
        lr_d = args.gan_lr_d if args.gan_lr_d is not None else args.lr
        optG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
        optD = torch.optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))
        bce = torch.nn.BCEWithLogitsLoss()
        history = []
        epoch_bar = tqdm(range(1, args.epochs + 1), desc=f'[TRAIN] {args.model}', unit='epoch')
        for epoch in epoch_bar:
            G.train()
            D.train()
            dl = gl = 0.0
            for signal, _time, label, participant in train_loader:
                signal = signal.to(device)
                label = label.long().to(device)
                participant = participant.long().to(device)
                B = signal.size(0)
                p_in = participant.clone()
                if args.part_dropout > 0:                                    # train the null-token path for LOSO
                    p_in[torch.rand(B, device=device) < args.part_dropout] = G.null_part_idx
                # D step
                z = torch.randn(B, z_dim, device=device)
                fake = G(z, label, p_in).detach()
                d_real = D(signal, label)
                d_fake = D(fake, label)
                if args.gan_loss == 'hinge':
                    lossD = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()
                else:
                    lossD = bce(d_real, torch.full_like(d_real, 0.9)) + bce(d_fake, torch.zeros_like(d_fake))
                optD.zero_grad()
                lossD.backward()
                optD.step()
                # G step
                z = torch.randn(B, z_dim, device=device)
                g_logit = D(G(z, label, p_in), label)
                lossG = -g_logit.mean() if args.gan_loss == 'hinge' else bce(g_logit, torch.ones((B, 1), device=device))
                optG.zero_grad()
                lossG.backward()
                optG.step()
                dl += lossD.item()
                gl += lossG.item()
            n = len(train_loader)
            torch.save({'epoch': epoch, 'model_state': G.state_dict(), 'stats': train_ds.stats,
                        'embed_dim': args.embed_dim, 'part_embed_dim': args.part_embed_dim, 'z_dim': z_dim,
                        'hidden': args.gan_hidden,
                        'latent_dim': args.latent_dim, 'region': args.region, 'condition_on_participant': True,
                        'num_participants': num_participants, 'part_dropout': args.part_dropout, 'cv_mode': args.cv_mode}, ckpt_path)
            history.append({'epoch': epoch, 'lossD': dl / n, 'lossG': gl / n})
            epoch_bar.set_postfix({'D': f'{dl / n:.3f}', 'G': f'{gl / n:.3f}'})

    # save training history
    with open(history_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(f'[TRAIN] History saved to {history_path}\n[TRAIN] Checkpoint saved to {ckpt_path}')

if __name__ == '__main__':
    main()
