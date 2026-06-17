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

    def _sensor_kernel(self, log_tau_s=None):
        """Fixed sensor kernel (buffer) unless a learnable log_tau_s is supplied, in which case
        rebuild s(t)=(1/tau_s) e^(-t/tau_s) so the sensor lag can be corrected during training."""
        if log_tau_s is None:
            return self.s
        tau_s = torch.exp(log_tau_s)
        return torch.where(self.t > 0, (1.0 / tau_s) * torch.exp(-self.t / tau_s), torch.zeros_like(self.t))

    def compute_kernels(self, log_A, log_D, log_v, log_tau_s=None):
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

        # convolve with sensor (fixed buffer or learnable tau_s)
        s = self._sensor_kernel(log_tau_s)
        B = h_channel.shape[0]
        h_channel_in = h_channel.unsqueeze(0)
        padded = F.pad(h_channel_in, (self.T - 1, 0))
        s_kernel = s.flip(0).view(1, 1, -1).expand(B, 1, -1)
        h_full = F.conv1d(padded, s_kernel, groups=B)
        h_full = h_full.squeeze(0) * self.dt
        h_full = h_full[:, :self.T]

        # normalize to unit integral
        h_full = h_full / (h_full.sum(dim=1, keepdim=True) + 1e-8)

        return h_full

    def forward(self, u, log_A, log_D, log_v, log_tau_s=None, apply_softplus=True):
        # apply softplus (free source) or pass through (already-non-negative parametric source) + baseline mask
        u_clean = (F.softplus(u) if apply_softplus else u) * self.baseline_mask

        # per-sample kernels
        kernels = self.compute_kernels(log_A, log_D, log_v, log_tau_s)   # (B, T)

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

    def sample(self, n, y, device, return_aux: bool = False, participant: int | None = None):
        z = torch.randn(n, self.latent_dim, device=device)
        if y.dim() == 0:
            y = y.expand(n)
        if not self._cond_part:
            p = None
        elif participant is not None:  # e.g. null_part_idx for LOSO generation of an unseen subject
            p = torch.full((n,), participant, dtype=torch.long, device=device)
        else:
            p = torch.randint(0, self.num_participants, (n,), device=device)

        self.eval()
        with torch.no_grad():
            x_hat = self.decoder(z, y.to(device), p)
            if not return_aux:
                return x_hat

            u_raw = self.u_head(z)
            log_A, log_D, log_v = self._cir_params(z)
            humidity_phys, u_post_softplus = self.cir_conv(u_raw, log_A, log_D, log_v)
            return x_hat, u_post_softplus, (log_A, log_D, log_v), humidity_phys


class SharedTransportPINN(CVAE):
    """Physics-AS-decoder generative model with a shared advection-diffusion transport for BOTH
    channels (heat-mass / Lewis analogy: exhaled heat and water-vapour ride the same turbulent
    airflow, so they share the transport (D, v) and differ only in their per-channel source u).

    Unlike `PhysicsInformedCVAE` (where physics was a side-penalty that the sampler *discarded*),
    here the physics IS the generative output: x_hat = [u_H * h(D,v), u_T * h(D,v)]. The transport
    kernel is unit-integral normalised so amplitude lives entirely in the (non-negative, rise-from-0)
    sources u_H, u_T; the shared, subject-invariant (D, v) is the regulariser that should transfer
    across unseen subjects (LOSO). The conv decoder of the base CVAE is inherited (encoder + the
    null-token machinery come with it) but its conv stack is unused — only its label/part embeddings
    are reused to condition the sources."""

    def __init__(self, cir_params_init, t_grid, tau_s: float = 15.0, d0: float = 0.03,
                 baseline_samples: int = 5, learn_transport: bool = False, residual: bool = False,
                 class_transport: bool = False, parametric_source: bool = False, **cvae_kwargs):
        super().__init__(**cvae_kwargs)
        rng_state = torch.get_rng_state()
        self.cir_conv = CIRConvolution(t_grid, tau_s=tau_s, d0=d0, baseline_samples=baseline_samples)
        T = len(t_grid)
        self.learn_transport = learn_transport
        # class_transport: per-class learnable (D,v) + a learnable global sensor lag tau_s, so the
        # physics expresses the breath class through interpretable transport AND can correct the
        # diagnosed over-smoothing (fixed tau_s=15s is ~2.5x too sluggish for the data's transient).
        self.class_transport = class_transport
        # parametric_source: replace the free 36-dim source MLPs with an interpretable analytic
        # injection u(t)=A*sigmoid((t-t0)/rate) whose params (onset t0, ramp rate, per-channel
        # amplitude) are PER-CLASS (deterministic base, indexed by y) + a small z perturbation. This
        # removes the free-source degeneracy (which made transport unidentifiable) so the class flows
        # through identifiable physics and prior-sampling is class-faithful. Transport is global+learnable.
        self.parametric_source = parametric_source
        if parametric_source:
            self.src_base = nn.Embedding(self.num_classes, 4)   # [t0_raw, rate_raw, A_H_raw, A_T_raw] per class
            self.src_pert = nn.Linear(self.latent_dim, 4)       # within-class variation from z
            nn.init.zeros_(self.src_base.weight)
            nn.init.zeros_(self.src_pert.weight); nn.init.zeros_(self.src_pert.bias)
        # residual=True: add the inherited conv decoder as a learned residual on top of the physics
        # envelope (x_hat = physics + decoder), to recover the class-discriminative capacity the
        # pure-physics decoder lacks while keeping the shared-transport physics as the base.
        self.residual = residual

        # sources conditioned on [z, class_embed, participant_embed] (reuse the decoder's embeddings
        # so the null-token expansion done in CVAE.__init__ is shared for LOSO generation)
        emb = self.decoder.label_embed.embedding_dim
        cond_dim = self.latent_dim + emb + (self.decoder.part_embed.embedding_dim if self._cond_part else 0)
        self.uH_head = nn.Sequential(nn.Linear(cond_dim, 64), nn.ReLU(), nn.Linear(64, T))
        self.uT_head = nn.Sequential(nn.Linear(cond_dim, 64), nn.ReLU(), nn.Linear(64, T))

        _, D_init, v_init, _ = cir_params_init
        log_D_init = float(np.log(D_init))
        log_v_init = float(np.log(v_init))
        if class_transport:
            # per-class transport (indexed by y) + learnable global sensor lag
            self.log_D = nn.Parameter(torch.full((self.num_classes,), log_D_init, dtype=torch.float32))
            self.log_v = nn.Parameter(torch.full((self.num_classes,), log_v_init, dtype=torch.float32))
            self.log_tau_s = nn.Parameter(torch.tensor(float(np.log(tau_s)), dtype=torch.float32))
        elif parametric_source:
            # global learnable transport (now identifiable because the source is constrained)
            self.log_D = nn.Parameter(torch.tensor(log_D_init, dtype=torch.float32))
            self.log_v = nn.Parameter(torch.tensor(log_v_init, dtype=torch.float32))
            self.log_tau_s = nn.Parameter(torch.tensor(float(np.log(tau_s)), dtype=torch.float32))
        elif learn_transport:
            self.log_D = nn.Parameter(torch.tensor(log_D_init, dtype=torch.float32))
            self.log_v = nn.Parameter(torch.tensor(log_v_init, dtype=torch.float32))
        else:
            self.register_buffer("log_D", torch.tensor(log_D_init, dtype=torch.float32))
            self.register_buffer("log_v", torch.tensor(log_v_init, dtype=torch.float32))
        torch.set_rng_state(rng_state)

    def _cond(self, z, y, p):
        parts = [z, self.decoder.label_embed(y)]
        if self._cond_part and p is not None:
            parts.append(self.decoder.part_embed(p))
        return torch.cat(parts, dim=1)

    def _physics(self, z, y, p):
        B = z.shape[0]
        zero_A = torch.zeros(B, device=z.device)          # A is normalised out of the unit-integral kernel
        # --- transport params ---
        if self.class_transport:
            log_D = self.log_D[y]                          # per-class transport, indexed by breath class
            log_v = self.log_v[y]
            log_tau_s = self.log_tau_s                     # learnable global sensor lag
        elif self.parametric_source:
            log_D = self.log_D.expand(B)
            log_v = self.log_v.expand(B)
            log_tau_s = self.log_tau_s
        else:
            log_D = self.log_D.expand(B)
            log_v = self.log_v.expand(B)
            log_tau_s = None
        # --- sources ---
        if self.parametric_source:
            raw = self.src_base(y) + 0.1 * self.src_pert(z)            # per-class base + small z variation
            T = self.cir_conv.T
            idx = torch.arange(T, device=z.device, dtype=torch.float32).unsqueeze(0)   # (1,T) sample index
            t0 = torch.sigmoid(raw[:, 0:1]) * T                        # onset in [0,T] samples
            rate = F.softplus(raw[:, 1:2]) + 0.5                       # ramp width
            A_H = F.softplus(raw[:, 2:3]); A_T = F.softplus(raw[:, 3:4])
            uH_src = A_H * torch.sigmoid((idx - t0) / rate)            # interpretable analytic injection
            uT_src = A_T * torch.sigmoid((idx - t0) / rate)
            humidity_phys, uH = self.cir_conv(uH_src, zero_A, log_D, log_v, log_tau_s, apply_softplus=False)
            temperature_phys, uT = self.cir_conv(uT_src, zero_A, log_D, log_v, log_tau_s, apply_softplus=False)
        else:
            c = self._cond(z, y, p)
            humidity_phys, uH = self.cir_conv(self.uH_head(c), zero_A, log_D, log_v, log_tau_s)
            temperature_phys, uT = self.cir_conv(self.uT_head(c), zero_A, log_D, log_v, log_tau_s)
        x_hat = torch.stack([humidity_phys, temperature_phys], dim=1)
        if self.residual:
            x_hat = x_hat + self.decoder(z, y, p)   # class-discriminative residual on the physics envelope
        return x_hat, uH, uT, (zero_A, log_D, log_v)

    def forward(self, x, y, p=None):
        # CFG-style participant dropout (reuse CVAE's null-token logic), then encode → z → physics
        if self.training and self._cond_part and p is not None and self.part_dropout > 0.0:
            self._total_steps += 1
            if torch.rand(1).item() < self.part_dropout:
                p = torch.full_like(p, self.null_part_idx)
                self._null_steps += 1
        mu, logvar = self.encoder(x, y, p)
        z = self.reparameterize(mu, logvar)
        x_hat, uH, _uT, cir = self._physics(z, y, p)
        # 6-tuple matching PhysicsInformedCVAE so the existing pinn train loop is reused (lambda_phys=0):
        # (x_hat, mu, logvar, u_post_softplus, cir_params, humidity_phys)
        return x_hat, mu, logvar, uH, cir, x_hat[:, 0, :]

    def sample(self, n, y, device, return_aux: bool = False, participant: int | None = None):
        z = torch.randn(n, self.latent_dim, device=device)
        if y.dim() == 0:
            y = y.expand(n)
        if not self._cond_part:
            p = None
        elif participant is not None:  # null_part_idx for LOSO generation of an unseen subject
            p = torch.full((n,), participant, dtype=torch.long, device=device)
        else:
            p = torch.randint(0, self.num_participants, (n,), device=device)
        self.eval()
        with torch.no_grad():
            x_hat, uH, uT, cir = self._physics(z, y.to(device), p)   # generation EMITS physics
        if return_aux:
            return x_hat, uH, uT, cir
        return x_hat
