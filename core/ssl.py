import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon
from torch import nn

from core.data import load_dataset, n_loso_folds


class SSLEncoder(nn.Module):
    """1D-conv encoder -> embedding, with a projection head used only during contrastive pretraining."""
    def __init__(self, emb_dim: int = 32) -> None:
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(2, 32, 3, padding=1), nn.ReLU(), nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.emb = nn.Linear(64, emb_dim)
        self.proj = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, emb_dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e = self.emb(self.conv(x).flatten(1))
        return e, self.proj(e)

def _augment(x: torch.Tensor) -> torch.Tensor:
    """Two-view augmentation for contrastive learning: jitter + per-channel scaling + a random time-mask."""
    # jitter + per-channel scaling
    x = x + 0.1 * torch.randn_like(x)
    x = x * (1 + 0.1 * torch.randn(x.shape[0], x.shape[1], 1, device=x.device))
    # random time-mask
    B, _C, T = x.shape
    starts = torch.randint(0, T - 5, (B,))
    mask = torch.ones(B, 1, T, device=x.device)
    for b in range(B):
        mask[b, 0, starts[b]:starts[b] + 5] = 0.0
    return x * mask

def _nt_xent(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    """Standard SimCLR NT-Xent contrastive loss over a batch of two augmented views."""
    B = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    sim = z @ z.t() / tau
    sim.fill_diagonal_(-1e9)
    targets = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    return F.cross_entropy(sim, targets)

def _baseline_correct(sig: np.ndarray, nb: int = 5) -> np.ndarray:
    """Baseline-correct a (2, T) signal (subtract per-channel pre-onset mean) to match the classifier feature space."""
    out = sig.astype(np.float32).copy()
    out[0] -= out[0, :nb].mean()
    out[1] -= out[1, :nb].mean()
    return out

def load_all_signals(dataset_dir: str = 'dataset') -> np.ndarray:
    """All trials (both regions), baseline-corrected, as an (N, 2, T) array for unlabeled pretraining."""
    df = load_dataset(dataset_dir)
    return np.stack([_baseline_correct(np.stack([r['humidity'], r['temperature']])) for _, r in df.iterrows()])

def embed(encoder: SSLEncoder, signals: np.ndarray, device: torch.device | None = None) -> np.ndarray:
    """Return (N, emb_dim) embeddings for (N, 2, T) baseline-corrected signals (eval mode, no grad)."""
    device = device or torch.device('cpu')
    encoder.eval()
    with torch.no_grad():
        e, _ = encoder(torch.tensor(signals, dtype=torch.float32, device=device))
    return e.cpu().numpy()

def run_pretrain(args: argparse.Namespace) -> None:
    """Pretrain the contrastive encoder on all trials and save it."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    device = torch.device('cpu')
    # load data + build encoder
    X = torch.tensor(load_all_signals(), dtype=torch.float32)
    print(f'[SSL] pretraining on {X.shape[0]} trials, {args.epochs} epochs')
    enc = SSLEncoder(args.emb_dim).to(device)
    opt = torch.optim.Adam(enc.parameters(), lr=args.lr)
    N = X.shape[0]
    # contrastive training loop
    for ep in range(args.epochs):
        perm = torch.randperm(N)
        tot = 0.0
        for i in range(0, N, args.batch_size):
            xb = X[perm[i:i + args.batch_size]].to(device)
            if xb.shape[0] < 4:
                continue
            _, p1 = enc(_augment(xb))
            _, p2 = enc(_augment(xb))
            loss = _nt_xent(p1, p2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        if (ep + 1) % 50 == 0:
            print(f'[SSL] epoch {ep + 1}: loss {tot / max(1, N // args.batch_size):.4f}')
    # save encoder
    torch.save({'state': enc.state_dict(), 'emb_dim': args.emb_dim}, args.out)
    print(f'[SSL] saved encoder -> {args.out}')

def run_eval(args: argparse.Namespace) -> None:
    """TRTR ablation: does appending the ssl embeddings to the top-20 tsfresh features help the classifier?"""
    from core.tstr import trtr  # lazy: core.tstr imports core.ssl
    seeds = [int(s) for s in args.seeds.split(',')]
    nf = n_loso_folds(load_dataset('dataset')) if args.cv_mode == 'loso' else 5
    # trtr with vs without the ssl embeddings, per seed x fold
    base, ssl = [], []
    for seed in seeds:
        for fold in range(1, nf + 1):
            kw = {'dataset_dir': 'dataset', 'region': args.region, 'n_jobs': args.n_jobs, 'init_seed': seed,
                  'split_seed': args.split_seed, 'fold': fold, 'n_folds': nf, 'cv_mode': args.cv_mode, 'preprocessing': 'baseline'}
            b = trtr(**kw)['trtr_metrics']['accuracy']
            s = trtr(**kw, ssl_ckpt=args.ssl_ckpt)['trtr_metrics']['accuracy']
            base.append(b)
            ssl.append(s)
            print(f'{args.region} {args.cv_mode} seed{seed} fold{fold}: base={b:.3f} ssl={s:.3f} d{s - b:+.3f}', flush=True)
    # paired test + summary
    base, ssl = np.array(base), np.array(ssl)
    pv = wilcoxon(ssl, base).pvalue if np.any(ssl != base) else float('nan')
    print(f'{args.region} {args.cv_mode}: TRTR base={base.mean():.3f} ssl={ssl.mean():.3f} d{ssl.mean() - base.mean():+.3f} p={pv:.4f} (n={len(base)})', flush=True)
    if args.out:
        with open(args.out, 'w') as f:
            json.dump({'region': args.region, 'cv_mode': args.cv_mode, 'base': base.tolist(), 'ssl': ssl.tolist(), 'base_mean': float(base.mean()), 'ssl_mean': float(ssl.mean()), 'p': float(pv)}, f)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    # pretrain
    pt = sub.add_parser('pretrain')
    pt.add_argument('--epochs', type=int, default=300)
    pt.add_argument('--emb_dim', type=int, default=32)
    pt.add_argument('--batch_size', type=int, default=256)
    pt.add_argument('--lr', type=float, default=1e-3)
    pt.add_argument('--seed', type=int, default=0)
    pt.add_argument('--out', type=str, default='results/ssl/ssl_encoder.pt')
    # eval
    ev = sub.add_parser('eval')
    ev.add_argument('--ssl_ckpt', required=True)
    ev.add_argument('--region', required=True, choices=['mouth', 'nose'])
    ev.add_argument('--cv_mode', default='kfold', choices=['kfold', 'loso'])
    ev.add_argument('--seeds', default='0,1,7,42,123')
    ev.add_argument('--split_seed', type=int, default=42)
    ev.add_argument('--n_jobs', type=int, default=4)
    ev.add_argument('--out', default='')
    return p.parse_args()

def main() -> None:
    args = parse_args()
    if args.cmd == 'pretrain':
        run_pretrain(args)
    else:
        run_eval(args)

if __name__ == '__main__':
    main()
