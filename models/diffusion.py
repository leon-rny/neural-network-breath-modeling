import math

import torch
import torch.nn.functional as F
from torch import nn


def _timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal timestep embedding.

    :param t: Tensor of shape (B,) containing the timesteps.
    :param dim: The dimension of the output embedding.
    :return: Tensor of shape (B, dim) containing the timestep embeddings.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1))
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

class _Denoiser(nn.Module):
    """1D conv denoiser eps_theta(x_t, t, class[, participant]). Class/participant/time are injected as a per-step FiLM-like additive bias into each conv block."""
    def __init__(self, channels=2, num_classes=3, embed_dim=32, num_participants=5, part_embed_dim=8, cond_part=True, hidden=64, tdim=64, n_blocks=4):
        """
        :param channels: Number of signal channels (default 2 for flow and volume).
        :param num_classes: Number of classes to condition on (default 3).
        :param embed_dim: Dimension of the class embedding (default 32).
        :param num_participants: Number of participants to condition on (default 5).
        :param part_embed_dim: Dimension of the participant embedding (default 8).
        :param cond_part: Whether to condition on participant ID (default True).
        :param hidden: Hidden dimension size for the conv blocks (default 64).
        :param tdim: Dimension of the timestep embedding (default 64).
        :param n_blocks: Number of conv blocks (default 4).
        """
        super().__init__()
        self._cond_part = cond_part
        self.tdim = tdim
        self.label_embed = nn.Embedding(num_classes + 1, embed_dim)  # +1 row = null class (CFG uncond)
        self.part_embed = nn.Embedding(num_participants + 1, part_embed_dim) if cond_part else None
        self.null_part_idx = num_participants
        self.tproj = nn.Sequential(nn.Linear(tdim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        cond_in = embed_dim + (part_embed_dim if cond_part else 0)
        self.cproj = nn.Sequential(nn.Linear(cond_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.in_conv = nn.Conv1d(channels, hidden, 3, padding=1)
        self.blocks = nn.ModuleList([nn.Conv1d(hidden, hidden, 3, padding=1) for _ in range(n_blocks)])
        self.out_conv = nn.Conv1d(hidden, channels, 3, padding=1)

    def forward(self, x, t, y, p=None):
        """
        :param x: Tensor of shape (B, channels, n_samples) containing the noisy input signal at timestep t.
        :param t: Tensor of shape (B,) containing the timesteps.
        :param y: Tensor of shape (B,) containing the class labels to condition on.
        :param p: Optional tensor of shape (B,) containing the participant labels to condition on. If None, no participant conditioning will be applied.
        :return: Tensor of shape (B, channels, n_samples) containing the predicted noise.
        """
        h = self.in_conv(x)
        cond = self.tproj(_timestep_embedding(t, self.tdim))
        parts = [self.label_embed(y)]
        if self._cond_part and p is not None:
            parts.append(self.part_embed(p))
        cond = cond + self.cproj(torch.cat(parts, dim=1))
        cond = cond[:, :, None]  # (B, hidden, 1) broadcast over time
        for blk in self.blocks:
            h = F.silu(blk(h) + cond)
        return self.out_conv(h)

class ConditionalDiffusion(nn.Module):
    """Conditional 1D DDPM generator for breath signals (B, 2, 36). Trains eps-prediction on the
    z-scored signal (same normalization as the CVAE), conditions on class (+ participant), and
    supports null-token generation for LOSO (set part_dropout>0 so the null token is trained)."""
    def __init__(self, channels=2, n_samples=36, num_classes=3, num_participants=5, embed_dim=32, part_embed_dim=8, condition_on_participant=True, part_dropout=0.0, n_steps=200, hidden=64, cfg_dropout=0.1, guidance_scale=3.0):
        """
        :param channels: Number of signal channels (default 2 for flow and volume).
        :param n_samples: Number of time samples in the signal (default 36).
        :param num_classes: Number of classes to condition on (default 3).
        :param num_participants: Number of participants to condition on (default 5).
        :param embed_dim: Dimension of the class embedding (default 32).
        :param part_embed_dim: Dimension of the participant embedding (default 8).
        :param condition_on_participant: Whether to condition on participant ID (default True).
        :param part_dropout: Probability of dropping participant conditioning to the null token during training (default 0.0).
        :param n_steps: Number of diffusion steps (default 200).
        :param hidden: Hidden dimension size for the denoiser network (default 64).
        :param cfg_dropout: Probability of dropping class conditioning to the null token during training for classifier-free guidance (default 0.1).
        :param guidance_scale: Guidance scale (w) for classifier-free guidance during sampling (default 3.0). Higher values increase fidelity to the class condition at the cost of diversity. Set to 1.0 to disable CFG.
        """
        super().__init__()
        self._cond_part = condition_on_participant
        self.num_participants = num_participants
        self.null_part_idx = num_participants
        self.part_dropout = part_dropout
        self.num_classes = num_classes
        self.null_class_idx = num_classes  # classifier-free guidance: null/uncond class token
        self.cfg_dropout = cfg_dropout  # prob of dropping class to null during training
        self.guidance_scale = guidance_scale  # w at sampling: eps = eps_uncond + w*(eps_cond-eps_uncond)
        self.n_steps = n_steps
        self.n_samples = n_samples
        self.channels = channels
        self.net = _Denoiser(channels, num_classes, embed_dim, num_participants, part_embed_dim, condition_on_participant, hidden)
        betas = torch.linspace(1e-4, 0.02, n_steps)  # linear schedule
        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', acp)
        self.register_buffer('sqrt_acp', acp.sqrt())
        self.register_buffer('sqrt_1macp', (1.0 - acp).sqrt())

    def loss(self, x, y, p=None):
        """DDPM eps-prediction loss. Optional CFG-style participant dropout to the null token.
        
        :param x: Tensor of shape (B, channels, n_samples) containing the clean signals.
        :param y: Tensor of shape (B,) containing the class labels to condition on.
        :param p: Optional tensor of shape (B,) containing the participant labels to condition on. If None, no participant conditioning will be applied.
        :return: Scalar tensor containing the MSE loss between the predicted noise and the true noise.
        """
        B = x.shape[0]
        if self.training and self._cond_part and p is not None and self.part_dropout > 0.0 and torch.rand(1).item() < self.part_dropout:
            p = torch.full_like(p, self.null_part_idx)
        if self.training and self.cfg_dropout > 0.0 and torch.rand(1).item() < self.cfg_dropout:
            y = torch.full_like(y, self.null_class_idx)  # CFG: train the unconditional path too
        t = torch.randint(0, self.n_steps, (B,), device=x.device)
        noise = torch.randn_like(x)
        x_t = self.sqrt_acp[t][:, None, None] * x + self.sqrt_1macp[t][:, None, None] * noise
        pred = self.net(x_t, t, y, p)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(self, n, y, device, participant: int | None = None):
        """Standard DDPM ancestral sampling. Returns (n, channels, n_samples) in z-scored space.
        
        :param n: Number of samples to generate.
        :param y: Tensor of shape (n,) or scalar containing the class labels to condition on.
        :param device: The device to generate the samples on.
        :param participant: Optional int specifying the participant to condition on. If None, a random participant will be sampled for each generated sample. Use null_part_idx for LOSO unseen-subject gen.
        :return: Tensor of shape (n, channels, n_samples) containing the generated samples
        """
        if y.dim() == 0:
            y = y.expand(n)
        y = y.to(device)
        if not self._cond_part:
            p = None
        elif participant is not None:  # null_part_idx for LOSO unseen-subject gen
            p = torch.full((n,), participant, dtype=torch.long, device=device)
        else:
            p = torch.randint(0, self.num_participants, (n,), device=device)
        self.eval()
        x = torch.randn(n, self.channels, self.n_samples, device=device)
        y_null = torch.full_like(y, self.null_class_idx)
        w = self.guidance_scale
        for i in reversed(range(self.n_steps)):
            t = torch.full((n,), i, dtype=torch.long, device=device)
            eps = self.net(x, t, y, p)
            if w != 1.0:  # classifier-free guidance
                eps_u = self.net(x, t, y_null, p)
                eps = eps_u + w * (eps - eps_u)
            x = (1.0 / self.alphas[i].sqrt()) * (x - (self.betas[i] / self.sqrt_1macp[i]) * eps)
            if i > 0:
                x = x + self.betas[i].sqrt() * torch.randn_like(x)
        return x
