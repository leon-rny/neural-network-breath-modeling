"""Self-supervised contrastive pretraining (SimCLR/NT-Xent) for breath signals (research idea #4).

Pretrains a small 1D-conv encoder on ALL trials (both regions pooled, no labels), so it learns a
representation less tied to any single subject. The embeddings are appended to the tsfresh features
for the downstream classifier (see core/tstr.py --ssl_ckpt). Standalone: `python -m core.ssl --epochs 300`.
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.data import load_dataset


class SSLEncoder(nn.Module):
    """1D-conv encoder -> embedding, with a projection head used only during contrastive pretraining."""
    def __init__(self, emb_dim: int = 32):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(2, 32, 3, padding=1), nn.ReLU(),
                                  nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),
                                  nn.AdaptiveAvgPool1d(1))
        self.emb = nn.Linear(64, emb_dim)
        self.proj = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, emb_dim))

    def forward(self, x):
        e = self.emb(self.conv(x).flatten(1))
        return e, self.proj(e)


def _augment(x: torch.Tensor) -> torch.Tensor:
    """Two-view augmentation for contrastive learning: jitter + per-channel scaling + a random time-mask."""
    x = x + 0.1 * torch.randn_like(x)
    x = x * (1 + 0.1 * torch.randn(x.shape[0], x.shape[1], 1, device=x.device))
    B, C, T = x.shape
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
    out[0] -= out[0, :nb].mean(); out[1] -= out[1, :nb].mean()
    return out


def load_all_signals(dataset_dir: str = 'dataset') -> np.ndarray:
    """All trials (both regions), baseline-corrected, as an (N, 2, T) array for unlabeled pretraining."""
    df = load_dataset(dataset_dir)
    return np.stack([_baseline_correct(np.stack([r['humidity'], r['temperature']])) for _, r in df.iterrows()])


def embed(encoder: SSLEncoder, signals: np.ndarray, device=None) -> np.ndarray:
    """Return (N, emb_dim) embeddings for (N, 2, T) baseline-corrected signals (eval mode, no grad)."""
    device = device or torch.device('cpu')
    encoder.eval()
    with torch.no_grad():
        e, _ = encoder(torch.tensor(signals, dtype=torch.float32, device=device))
    return e.cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--emb_dim', type=int, default=32)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=str, default='results/ssl/ssl_encoder.pt')
    args = p.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    import os; os.makedirs(os.path.dirname(args.out), exist_ok=True)
    device = torch.device('cpu')
    X = torch.tensor(load_all_signals(), dtype=torch.float32)
    print(f'[SSL] pretraining on {X.shape[0]} trials, {args.epochs} epochs')
    enc = SSLEncoder(args.emb_dim).to(device)
    opt = torch.optim.Adam(enc.parameters(), lr=args.lr)
    N = X.shape[0]
    for ep in range(args.epochs):
        perm = torch.randperm(N)
        tot = 0.0
        for i in range(0, N, args.batch_size):
            xb = X[perm[i:i + args.batch_size]].to(device)
            if xb.shape[0] < 4:
                continue
            _, p1 = enc(_augment(xb)); _, p2 = enc(_augment(xb))
            loss = _nt_xent(p1, p2)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        if (ep + 1) % 50 == 0:
            print(f'[SSL] epoch {ep + 1}: loss {tot / max(1, N // args.batch_size):.4f}')
    torch.save({'state': enc.state_dict(), 'emb_dim': args.emb_dim}, args.out)
    print(f'[SSL] saved encoder -> {args.out}')


if __name__ == '__main__':
    main()
