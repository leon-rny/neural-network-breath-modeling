import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vae import CVAE

# physics cvae
class CIRConvolution(nn.Module):
    def __init__(self, params, t_grid, tau_s=15.0, d0=0.03):
        super().__init__()

        # compute kernel
        A, D, v, _ = params
        h = self._compute_kernel(t_grid, A, D, v, tau_s, d0)
        h = h / h.sum()
        h_flipped = np.flip(h).copy()
        kernel_tensor = torch.tensor(h_flipped, dtype=torch.float32).view(1, 1, -1)
        self.register_buffer("kernel", kernel_tensor)
        self.kernel_length = h.shape[0]

        # learnable baseline
        self.baseline = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _compute_kernel(t_grid, A, D, v, tau_s, d0):
        t = np.asarray(t_grid, dtype=float)
        dt = t[1] - t[0]
        mask = t > 1e-9
        tt = t[mask]
        h_channel = np.zeros_like(t)
        h_channel[mask] = (A / np.sqrt(4.0 * np.pi * D * tt)) * np.exp(-((d0 - v * tt) ** 2) / (4.0 * D * tt))
        s_sensor = np.where(t > 0, (1.0 / tau_s) * np.exp(-t / tau_s), 0.0)
        return np.convolve(h_channel, s_sensor, mode='full')[:len(t)] * dt

    def forward(self, u):
        u = u.unsqueeze(1)
        u = F.softplus(u)
        u_padded = F.pad(u, (self.kernel_length - 1, 0))
        y = F.conv1d(u_padded, self.kernel)

        # # add saturation: maybe use a smooth clip at 1.0?!?
        # y = torch.tanh(y / 0.7) * 0.7

        return y.squeeze(1) + self.baseline

class PhysicsInformedCVAE(CVAE):
    def __init__(self, cir_params, t_grid, tau_s=15.0, **cvae_kwargs):
        super().__init__(**cvae_kwargs)
        self.cir_conv = CIRConvolution(cir_params, t_grid, tau_s=tau_s)

    def _estimate_onset(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, T) normalized humidity — mirrors PhysicsInformedDataset heuristic
        baseline_mean = h[:, :5].mean(dim=1, keepdim=True)
        baseline_std = h[:, :5].std(dim=1, keepdim=True)
        threshold = baseline_mean + 3.0 * baseline_std
        above = h > threshold
        has_onset = above.any(dim=1)
        onset = above.long().argmax(dim=1)
        onset[~has_onset] = 5
        return onset

    def forward(self, x, y, p=None):
        mu, logvar = self.encoder(x, y, p)
        z = self.reparameterize(mu, logvar)
        raw_out = self.decoder(z, y, p)

        u_humidity = raw_out[:, 0, :]
        temperature = raw_out[:, 1, :]

        humidity = self.cir_conv(u_humidity)
        x_hat = torch.stack([humidity, temperature], dim=1)

        u_post_softplus = F.softplus(u_humidity)
        onset_idx = self._estimate_onset(x[:, 0, :])

        return x_hat, mu, logvar, u_post_softplus, onset_idx

    def sample(self, n, y, device):
        z = torch.randn(n, self.latent_dim, device=device)
        if y.dim() == 0:
            y = y.expand(n)
        p = torch.randint(0, self.num_participants, (n,), device=device) if self._cond_part else None
        self.eval()
        with torch.no_grad():
            raw_out = self.decoder(z, y.to(device), p)
            u_humidity = raw_out[:, 0, :]
            temperature = raw_out[:, 1, :]
            humidity = self.cir_conv(u_humidity)
            return torch.stack([humidity, temperature], dim=1)