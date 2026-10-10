"""PI-CVAE: a CVAE whose decoder emits the calibrated measurement chain (transport + sensor), with the conv decoder as
residual."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.vae import CVAE


# physics cvae
class CIRConvolution(nn.Module):
    """Differentiable advection-diffusion channel + first-order sensor lag: a per-sample source u(t) is convolved with
    the unit-integral kernel h_channel(D, v) * s(tau_s)."""

    def __init__(self, t_grid: np.ndarray, tau_s: float = 15.0, d0: float = 0.03, baseline_samples: int = 5) -> None:
        super().__init__()
        self.d0 = d0
        self.tau_s = tau_s
        # time grid
        t = torch.tensor(t_grid, dtype=torch.float32)
        self.register_buffer('t', t)
        self.dt = float(t[1] - t[0])
        self.T = len(t_grid)
        # fixed sensor kernel
        s = torch.where(t > 0, (1.0 / tau_s) * torch.exp(-t / tau_s), torch.zeros_like(t))
        self.register_buffer('s', s)
        # baseline mask: the source is zero over the pre-onset baseline
        mask = torch.ones(self.T)
        mask[:baseline_samples] = 0.0
        self.register_buffer('baseline_mask', mask)

    def compute_kernels(self, log_D: torch.Tensor, log_v: torch.Tensor) -> torch.Tensor:
        """(B, T) unit-integral channel kernels convolved with the sensor kernel, for (B,) log D and log v."""
        D = torch.exp(log_D).unsqueeze(1)
        v = torch.exp(log_v).unsqueeze(1)
        t = self.t.unsqueeze(0)
        # advection-diffusion
        t_pos = torch.clamp(t, min=1e-9)
        prefactor = 1.0 / torch.sqrt(4 * np.pi * D * t_pos)
        exponent = -(self.d0 - v * t_pos) ** 2 / (4 * D * t_pos)
        h_channel = prefactor * torch.exp(exponent)
        h_channel = torch.where(t > 1e-9, h_channel, torch.zeros_like(h_channel))
        # convolve with the sensor kernel
        B = h_channel.shape[0]
        h_channel_in = h_channel.unsqueeze(0)
        padded = F.pad(h_channel_in, (self.T - 1, 0))
        s_kernel = self.s.flip(0).view(1, 1, -1).expand(B, 1, -1)
        h_full = F.conv1d(padded, s_kernel, groups=B)
        h_full = h_full.squeeze(0) * self.dt
        h_full = h_full[:, :self.T]
        # normalize to unit integral
        h_full = h_full / (h_full.sum(dim=1, keepdim=True) + 1e-8)
        return h_full

    def forward(self, u: torch.Tensor, log_D: torch.Tensor, log_v: torch.Tensor) -> torch.Tensor:
        """(B, T) raw source (softplus, zero over the baseline) -> (B, T) observed signal."""
        u_clean = F.softplus(u) * self.baseline_mask
        kernels = self.compute_kernels(log_D, log_v)  # (B, T)
        # per-sample convolution (channels-as-batch trick for groups=B)
        B, T = u_clean.shape
        u_in = u_clean.unsqueeze(0)
        k = kernels.flip(1).unsqueeze(1)
        padded = F.pad(u_in, (T - 1, 0))
        return F.conv1d(padded, k, groups=B).squeeze(0)

class SharedTransportPINN(CVAE):
    """PI-CVAE: per channel a source network maps [z, class embedding, participant embedding] to a non-negative source,
    convolved with the fixed calibrated transport (D, v) + sensor kernel shared by both channels (heat-mass analogy);
    the conditional conv decoder is added as a residual.

    :param cir_params_init: (A, D, v, t_shift) of the calibration (core.calibrate); only D and v are used
    :param residual: add the conv decoder output to the physics output
    :param cvae_kwargs: forwarded to CVAE
    """

    def __init__(self, cir_params_init: np.ndarray, t_grid: np.ndarray, tau_s: float = 15.0, d0: float = 0.03, baseline_samples: int = 5, residual: bool = True, **cvae_kwargs) -> None:
        super().__init__(**cvae_kwargs)
        rng_state = torch.get_rng_state()
        self.cir_conv = CIRConvolution(t_grid, tau_s=tau_s, d0=d0, baseline_samples=baseline_samples)
        T = len(t_grid)
        self.residual = residual
        emb = self.decoder.label_embed.embedding_dim
        cond_dim = self.latent_dim + emb + (self.decoder.part_embed.embedding_dim if self._cond_part else 0)
        self.uH_head = nn.Sequential(nn.Linear(cond_dim, 64), nn.ReLU(), nn.Linear(64, T))
        self.uT_head = nn.Sequential(nn.Linear(cond_dim, 64), nn.ReLU(), nn.Linear(64, T))
        _, D_init, v_init, _ = cir_params_init
        self.register_buffer('log_D', torch.tensor(float(np.log(D_init)), dtype=torch.float32))
        self.register_buffer('log_v', torch.tensor(float(np.log(v_init)), dtype=torch.float32))
        torch.set_rng_state(rng_state)

    def _cond(self, z: torch.Tensor, y: torch.Tensor, p: torch.Tensor | None) -> torch.Tensor:
        parts = [z, self.decoder.label_embed(y)]
        if self._cond_part and p is not None:
            parts.append(self.decoder.part_embed(p))
        return torch.cat(parts, dim=1)

    def decode(self, z: torch.Tensor, y: torch.Tensor, p: torch.Tensor | None = None) -> torch.Tensor:
        """Sources u_H, u_T through the shared transport + sensor kernel, plus the residual.

        :return: (B, 2, 36) signals in the training space (baseline-corrected, stdscale)
        """
        B = z.shape[0]
        log_D = self.log_D.expand(B)
        log_v = self.log_v.expand(B)
        c = self._cond(z, y, p)
        humidity = self.cir_conv(self.uH_head(c), log_D, log_v)
        temperature = self.cir_conv(self.uT_head(c), log_D, log_v)
        x_hat = torch.stack([humidity, temperature], dim=1)
        if self.residual:
            x_hat = x_hat + self.decoder(z, y, p)  # class-discriminative residual on the physics envelope
        return x_hat
