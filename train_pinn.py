import argparse
import os
import json
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from core.data import load_dataset, split_dataset, PhysicsInformedDataset
from core.utils import seed_everything, make_generator
from core.train import beta_capped
from models.pinn import PhysicsInformedCVAE

# train pinn
LOSS_KEYS = ("total", "recon", "kl", "phys")
DIAG_KEYS = ("u_norm", "logA_mean", "logA_std", "logD_mean", "logD_std", "logv_mean", "logv_std")
METRIC_KEYS = LOSS_KEYS + DIAG_KEYS

def compute_losses(x, x_hat, mu, logvar, humidity_phys, beta, lambda_phys):
    recon = F.mse_loss(x_hat, x)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    phys = F.mse_loss(humidity_phys, x_hat[:, 0, :])

    total = recon + beta * kl + lambda_phys * phys
    return {"total": total, "recon": recon, "kl": kl, "phys": phys}

def diag_stats(u_post_softplus, cir_params):
    log_A, log_D, log_v = cir_params
    return {
        "u_norm": u_post_softplus.norm(dim=1).mean().item(),
        "logA_mean": log_A.mean().item(), "logA_std": log_A.std().item(),
        "logD_mean": log_D.mean().item(), "logD_std": log_D.std().item(),
        "logv_mean": log_v.mean().item(), "logv_std": log_v.std().item(),
    }

@torch.no_grad()
def evaluate(model, loader, beta, lambda_phys, device):
    model.eval()
    totals = {k: 0.0 for k in METRIC_KEYS}
    n_batches = 0

    for batch in loader:
        x, _t, y, p, _onset_idx = batch
        x = x.to(device)
        y = y.to(device)
        p = p.to(device)

        x_hat, mu, logvar, u_post_softplus, cir_params, humidity_phys = model(x, y, p)
        losses = compute_losses(x, x_hat, mu, logvar, humidity_phys, beta=beta, lambda_phys=lambda_phys)
        diag = diag_stats(u_post_softplus, cir_params)

        for k in LOSS_KEYS:
            totals[k] += losses[k].item()
        for k in DIAG_KEYS:
            totals[k] += diag[k]
        n_batches += 1

    return {k: v / n_batches for k, v in totals.items()}

def train_one_epoch(model, loader, optimizer, beta, lambda_phys, device):
    model.train()
    totals = {k: 0.0 for k in METRIC_KEYS}
    n_batches = 0

    for batch in loader:
        x, _t, y, p, _onset_idx = batch
        x = x.to(device)
        y = y.to(device)
        p = p.to(device)

        x_hat, mu, logvar, u_post_softplus, cir_params, humidity_phys = model(x, y, p)
        losses = compute_losses(x, x_hat, mu, logvar, humidity_phys, beta=beta, lambda_phys=lambda_phys)

        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()

        with torch.no_grad():
            diag = diag_stats(u_post_softplus, cir_params)

        for k in LOSS_KEYS:
            totals[k] += losses[k].item()
        for k in DIAG_KEYS:
            totals[k] += diag[k]
        n_batches += 1

    return {k: v / n_batches for k, v in totals.items()}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--region", required=True, choices=["mouth", "nose"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--lambda_phys", type=float, required=True)
    p.add_argument("--latent_dim", type=int, default=16)
    p.add_argument("--embed_dim", type=int, default=8)
    p.add_argument("--part_embed_dim", type=int, default=8)
    p.add_argument("--num_epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta_max", type=float, default=0.1)
    p.add_argument("--tau_s", type=float, default=15.0)
    p.add_argument("--learn_cir_params", action="store_true")
    # jittering augmentation (training only; defaults = off)
    p.add_argument("--alpha", type=float, default=0.0)
    p.add_argument("--n_copies", type=int, default=1)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cpu")
    os.makedirs("results/pinn", exist_ok=True)

    print(f'[TRAIN] Seed: {args.seed} | Model: PINN | Region: {args.region} | Device: {device} | Epochs: {args.num_epochs}')
    run_id = f"{args.region}_s{args.seed}_ld{args.latent_dim}_ed{args.embed_dim}_phys{args.lambda_phys}"
    if args.alpha > 0 and args.n_copies > 1:
        run_id += f"_a{args.alpha}_n{args.n_copies}"
    ckpt_path = f"results/pinn/{run_id}_checkpoint.pt"

    # reproducibility
    seed_everything(args.seed)
    g = make_generator(args.seed)

    # dataset
    df = load_dataset("./dataset")
    df = df[df["region"] == args.region].reset_index(drop=True)
    df_train, df_val, _ = split_dataset(df, random_state=args.seed)
    train_ds = PhysicsInformedDataset(df_train, alpha=args.alpha, n_copies=args.n_copies)
    val_ds = PhysicsInformedDataset(df_val, stats=train_ds.stats)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False, worker_init_fn=4, generator=g)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    if args.alpha > 0 and args.n_copies > 1:
        n_clean = len(df_train)
        print(f'[TRAIN] Jitter: alpha={args.alpha}, n_copies={args.n_copies} ({n_clean} → {len(train_ds)} samples)')

    # CIR parameters
    params_cir = np.load(f"results/pinn/params_{args.region}.npy")
    t_grid = np.arange(36) * 2.0

    model = PhysicsInformedCVAE(cir_params_init=params_cir,
                                t_grid=t_grid,
                                tau_s=args.tau_s,
                                learn_cir_params=args.learn_cir_params,
                                latent_dim=args.latent_dim,
                                num_classes=3,
                                embed_dim=args.embed_dim,
                                condition_on_participant=True,
                                num_participants=3,
                                part_embed_dim=args.part_embed_dim)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    history = {f"{split}_{k}": [] for split in ("train", "val") for k in METRIC_KEYS}
    history["beta"] = []

    best_val_recon = math.inf
    best_epoch = -1

    pbar = tqdm(range(args.num_epochs), desc="[TRAIN] PINN", unit="epoch")
    for epoch in pbar:
        beta = beta_capped(epoch, args.num_epochs // 2, args.beta_max)

        train_metrics = train_one_epoch(model, train_loader, optimizer, beta=beta, lambda_phys=args.lambda_phys, device=device)
        val_metrics = evaluate(model, val_loader, beta=beta, lambda_phys=args.lambda_phys, device=device)

        for k in METRIC_KEYS:
            history[f"train_{k}"].append(train_metrics[k])
            history[f"val_{k}"].append(val_metrics[k])
        history["beta"].append(beta)

        if val_metrics["recon"] < best_val_recon and beta >= args.beta_max:
            best_val_recon = val_metrics["recon"]
            best_epoch = epoch
            torch.save({"model_state": model.state_dict(),
                        "stats": train_ds.stats,
                        "latent_dim": args.latent_dim,
                        "embed_dim": args.embed_dim,
                        "part_embed_dim": args.part_embed_dim,
                        "region": args.region,
                        "cir_params": params_cir,
                        "t_grid": t_grid,
                        "tau_s": args.tau_s,
                        "learn_cir_params": args.learn_cir_params,
                        "condition_on_participant": True,
                        "num_participants": 3}, ckpt_path)

        pbar.set_postfix(b=f"{beta:.3f}",
                        tr=f"{train_metrics["recon"]:.4f}",
                        vr=f"{val_metrics["recon"]:.4f}",
                        kl=f"{val_metrics["kl"]:.4f}",
                        ph=f"{val_metrics["phys"]:.4f}",
                        u=f"{train_metrics["u_norm"]:.3f}")

    print(f"[TRAIN] Best validation recon: {best_val_recon:.4f} at epoch {best_epoch}.")

    history_path = f"results/pinn/{run_id}_history.json"
    with open(history_path, "w") as f:
        json.dump({"history": history, "best_epoch": best_epoch, "best_val_recon": best_val_recon}, f)
