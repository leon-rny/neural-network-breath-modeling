import os
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
@torch.no_grad()
def evaluate(model, loader, beta, lambda_sparsity, lambda_baseline, device):
    model.eval()
    totals = {"total": 0.0, "recon": 0.0, "kl": 0.0, "sparsity": 0.0, "baseline": 0.0}
    n_batches = 0

    for batch in loader:
        x, _t, y, p, _ = batch
        x = x.to(device)
        y = y.to(device)
        p = p.to(device)

        x_hat, mu, logvar, u_post_softplus, onset_idx = model(x, y, p)
        losses = compute_losses(x, x_hat, mu, logvar, u_post_softplus, onset_idx, beta=beta, lambda_sparsity=lambda_sparsity, lambda_baseline=lambda_baseline)

        for k in totals:
            totals[k] += losses[k].item()
        n_batches += 1

    return {k: v / n_batches for k, v in totals.items()}

def compute_losses(x, x_hat, mu, logvar, u_post_softplus, onset_idx, beta, lambda_sparsity, lambda_baseline):
    recon = F.mse_loss(x_hat, x)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    sparsity = u_post_softplus.abs().mean()

    # penalize any activity in the baseline window
    B, T = u_post_softplus.shape
    t_idx = torch.arange(T, device=u_post_softplus.device).unsqueeze(0)
    mask = (t_idx < onset_idx.unsqueeze(1)).float()
    baseline_penalty = (u_post_softplus * mask).abs().sum() / (mask.sum() + 1e-8)

    total = recon + beta * kl+ lambda_baseline * baseline_penalty# + lambda_sparsity * sparsity
    return {"total": total, "recon": recon, "kl": kl, "sparsity": sparsity, "baseline": baseline_penalty}

def train_one_epoch(model, loader, optimizer, beta, lambda_sparsity, lambda_baseline, device):
    model.train()
    totals = {"total": 0.0, "recon": 0.0, "kl": 0.0, "sparsity": 0.0, "baseline": 0.0}
    n_batches = 0

    for batch in loader:
        x, _t, y, p, onset_idx = batch
        x = x.to(device)
        y = y.to(device)
        p = p.to(device)

        x_hat, mu, logvar, u_post_softplus, onset_idx = model(x, y, p)

        losses = compute_losses(x, x_hat, mu, logvar, u_post_softplus, onset_idx, beta=beta, lambda_sparsity=lambda_sparsity, lambda_baseline=lambda_baseline)

        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()

        for k in totals:
            totals[k] += losses[k].item()
        n_batches += 1

    return {k: v / n_batches for k, v in totals.items()}

# params
device = torch.device('cuda' if torch.cuda.is_available() else
                      'mps'  if torch.backends.mps.is_available() else 'cpu')
REGION = ['nose'] # 'nose'
SEED = [0, 1, 7, 42, 123]
NUM_EPOCHS = 500
BATCH_SIZE = 16
LR = 1e-3 
BETA_MAX = 0.1
LAMBDA_SPARSITY = 0.001
LAMBDA_BASELINE = 1.0
TAU_S = 15.0
os.makedirs('results/pinn', exist_ok=True)

for seed in SEED:
    for region in REGION:
        run_id = f'{region}_s{seed}_ld16_ed8'
        ckpt_path = f'results/pinn/{run_id}_checkpoint.pt'

        # reproducibility
        seed_everything(seed)
        g = make_generator(seed)

        # dataset
        df = load_dataset('./dataset')
        df = df[df['region'] == region].reset_index(drop=True)
        df_train, df_val, _ = split_dataset(df, random_state=seed)
        train_ds = PhysicsInformedDataset(df_train)
        val_ds = PhysicsInformedDataset(df_val, stats=train_ds.stats)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, worker_init_fn=4, generator=g)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        # CIR parameters
        params_cir = np.load(f'results/pinn/params_{region}.npy')
        t_grid = np.arange(36) * 2.0

        model = PhysicsInformedCVAE(cir_params=params_cir,
                                    t_grid=t_grid,
                                    tau_s=TAU_S,
                                    latent_dim=16,
                                    num_classes=3,
                                    condition_on_participant=True,
                                    num_participants=3)
        model = model.to(device)
        optimizer = Adam(model.parameters(), lr=LR)

        history = {"train_total": [], "train_recon": [], "train_kl": [], "train_sparsity": [], "train_baseline": [],
                "val_total": [], "val_recon": [], "val_kl": [], "val_sparsity": [], "val_baseline": [], "beta": []}

        best_val_recon = math.inf
        best_epoch = -1

        pbar = tqdm(range(NUM_EPOCHS), desc="Training")
        for epoch in pbar:
            beta = beta_capped(epoch, NUM_EPOCHS, BETA_MAX)

            train_metrics = train_one_epoch(model, train_loader, optimizer, beta=beta, lambda_sparsity=LAMBDA_SPARSITY, lambda_baseline=LAMBDA_BASELINE, device=device)
            val_metrics = evaluate(model, val_loader, beta=beta, lambda_sparsity=LAMBDA_SPARSITY, lambda_baseline=LAMBDA_BASELINE, device=device)

            for k in ("total", "recon", "kl", "sparsity", "baseline"):
                history[f"train_{k}"].append(train_metrics[k])
                history[f"val_{k}"].append(val_metrics[k])
            history["beta"].append(beta)

            if val_metrics["recon"] < best_val_recon and beta_capped(epoch, NUM_EPOCHS, BETA_MAX) >= BETA_MAX:
                best_val_recon = val_metrics["recon"]
                best_epoch = epoch
                # save model
                torch.save({'model_state': model.state_dict(),
                            'stats': train_ds.stats,
                            'latent_dim': 16,
                            'embed_dim': 8,
                            'part_embed_dim': 8,
                            'region': region,
                            'cir_params': params_cir,
                            't_grid': t_grid,
                            'tau_s': TAU_S,
                            'condition_on_participant': True,
                            'num_participants': 3}, ckpt_path)

            pbar.set_postfix(b=f"{beta:.3f}",
                            tr=f"{train_metrics['recon']:.4f}",
                            vr=f"{val_metrics['recon']:.4f}",
                            kl=f"{val_metrics['kl']:.4f}",
                            bl=f"{val_metrics['baseline']:.4f}")

        print(f"\nBest validation recon: {best_val_recon:.4f} at epoch {best_epoch}")