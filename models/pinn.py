import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vae import CVAE

# physics cvae
class CIRConvolution(nn.Module):
    def __init__(self, t_grid: np.ndarray, tau_s: float = 15.0, d0: float = 0.03, baseline_samples: int = 5):
        super().__init__()
        self.d0 = d0
        self.tau_s = tau_s

        # time grid
        t = torch.tensor(t_grid, dtype=torch.float32)
        self.register_buffer("t", t)
        self.dt = float(t[1] - t[0])
        self.T = len(t_grid)

        # fixed sensor kernel
        s = torch.where(t > 0, (1.0 / tau_s) * torch.exp(-t / tau_s), torch.zeros_like(t))
        self.register_buffer("s", s)

        # baseline mask
        mask = torch.ones(self.T)
        mask[:baseline_samples] = 0.0
        self.register_buffer("baseline_mask", mask)

    def compute_kernels(self, log_A, log_D, log_v):
        A = torch.exp(log_A).unsqueeze(1)
        D = torch.exp(log_D).unsqueeze(1)
        v = torch.exp(log_v).unsqueeze(1)
        t = self.t.unsqueeze(0)

        # advection-diffusion
        t_pos = torch.clamp(t, min=1e-9)
        prefactor = A / torch.sqrt(4 * np.pi * D * t_pos)
        exponent = -(self.d0 - v * t_pos) ** 2 / (4 * D * t_pos)
        h_channel = prefactor * torch.exp(exponent)
        h_channel = torch.where(t > 1e-9, h_channel, torch.zeros_like(h_channel))

        # convolve with sensor
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

    def forward(self, u, log_A, log_D, log_v):
        # apply softplus + baseline mask
        u_clean = F.softplus(u) * self.baseline_mask

        # per-sample kernels
        kernels = self.compute_kernels(log_A, log_D, log_v)   # (B, T)

        # per-sample convolution (channels-as-batch trick for groups=B)
        B, T = u_clean.shape
        u_in = u_clean.unsqueeze(0)
        k = kernels.flip(1).unsqueeze(1)
        padded = F.pad(u_in, (T - 1, 0))
        y = F.conv1d(padded, k, groups=B).squeeze(0)

        return y, u_clean

class PhysicsInformedCVAE(CVAE):
    def __init__(self, cir_params_init, t_grid, tau_s: float = 15.0, d0: float = 0.03,
                 baseline_samples: int = 5, learn_cir_params: bool = True, **cvae_kwargs):
        super().__init__(**cvae_kwargs)
        # snapshot the RNG state the plain CVAE would leave, so the physics heads below
        # don't shift the reparameterize noise stream (restored at the end of __init__).
        rng_state = torch.get_rng_state()
        self.cir_conv = CIRConvolution(t_grid, tau_s=tau_s, d0=d0, baseline_samples=baseline_samples)
        T = len(t_grid)
        self.learn_cir_params = learn_cir_params

        # u_head
        self.u_head = nn.Sequential(nn.Linear(self.latent_dim, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, T))

        A_init, D_init, v_init, _ = cir_params_init
        log_A_init = float(np.log(A_init))
        log_D_init = float(np.log(D_init))
        log_v_init = float(np.log(v_init))

        if learn_cir_params:
            self.cir_param_head = nn.Sequential(nn.Linear(self.latent_dim, 32),
                                                nn.ReLU(),
                                                nn.Linear(32, 3))
            with torch.no_grad():
                self.cir_param_head[-1].bias.copy_(torch.tensor([log_A_init, log_D_init, log_v_init], dtype=torch.float32))
                self.cir_param_head[-1].weight.data *= 0.01
        else:
            self.register_buffer("log_A_fixed", torch.tensor(log_A_init, dtype=torch.float32))
            self.register_buffer("log_D_fixed", torch.tensor(log_D_init, dtype=torch.float32))
            self.register_buffer("log_v_fixed", torch.tensor(log_v_init, dtype=torch.float32))

        # restore so the global RNG state
        torch.set_rng_state(rng_state)

    def _cir_params(self, z):
        if self.learn_cir_params:
            log_A, log_D, log_v = self.cir_param_head(z).unbind(dim=1)
        else:
            B = z.shape[0]
            log_A = self.log_A_fixed.expand(B)
            log_D = self.log_D_fixed.expand(B)
            log_v = self.log_v_fixed.expand(B)
        return log_A, log_D, log_v

    def forward(self, x, y, p=None):
        mu, logvar = self.encoder(x, y, p)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z, y, p)

        u_raw = self.u_head(z)
        log_A, log_D, log_v = self._cir_params(z)
        humidity_phys, u_post_softplus = self.cir_conv(u_raw, log_A, log_D, log_v)

        return x_hat, mu, logvar, u_post_softplus, (log_A, log_D, log_v), humidity_phys

    def sample(self, n, y, device, return_aux: bool = False):
        z = torch.randn(n, self.latent_dim, device=device)
        if y.dim() == 0:
            y = y.expand(n)
        p = torch.randint(0, self.num_participants, (n,), device=device) if self._cond_part else None

        self.eval()
        with torch.no_grad():
            x_hat = self.decoder(z, y.to(device), p)
            if not return_aux:
                return x_hat

            u_raw = self.u_head(z)
            log_A, log_D, log_v = self._cir_params(z)
            humidity_phys, u_post_softplus = self.cir_conv(u_raw, log_A, log_D, log_v)
            return x_hat, u_post_softplus, (log_A, log_D, log_v), humidity_phys
