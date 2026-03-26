import torch
import torch.nn as nn

from models.vae import ConditionalEncoder

T_MAX = 75.0  # seconds — normalization constant for time input


class PhysicsParamHead(nn.Module):
    """
    Maps class embedding to 6 non-negative physics parameters for the saturation ODE:
      dH/dt_norm = α*(H_sat - H) - β*H
      dT/dt_norm = γ*(T_sat - T) - δ*T
    All parameters are in normalized-time units (t_norm = t / T_MAX).
    """
    def __init__(self, embed_dim: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(embed_dim, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 6),
                                 nn.Softplus())  # ensures strict positivity

    def forward(self, y_embed: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        :param y_embed: (B, embed_dim) class embeddings
        :return: 6 tensors of shape (B,): alpha, beta, H_sat, gamma, delta, T_sat
        """
        params = self.net(y_embed)  # (B, 6)
        return params.unbind(dim=1)  # 6 × (B,)


class PINNDecoder(nn.Module):
    """
    Continuous-time decoder: f_θ(t_norm, z, y_embed) → (H(t), T(t)).
    Tanh activations ensure smooth, twice-differentiable output needed for physics loss.
    """
    def __init__(self, latent_dim: int = 32, embed_dim: int = 16) -> None:
        super().__init__()
        in_dim = 1 + latent_dim + embed_dim
        self.net = nn.Sequential(nn.Linear(in_dim, 256), nn.Tanh(),
                                 nn.Linear(256, 256), nn.Tanh(),
                                 nn.Linear(256, 256), nn.Tanh(),
                                 nn.Linear(256, 2))

    def forward(self, t: torch.Tensor, z: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
        """
        :param t: (N, 1) normalized time values in [0, 1]
        :param z: (N, latent_dim) latent vectors
        :param y_embed: (N, embed_dim) class embeddings
        :return: (N, 2) [H(t), T(t)]
        """
        return self.net(torch.cat([t, z, y_embed], dim=1))


class PINNCVAE(nn.Module):
    """
    Physics-Informed Conditional VAE for breath signal generation.

    Encoder: ConditionalEncoder (same architecture as CVAE) maps (signal, class) → (mu, logvar).
    Decoder: continuous-time MLP evaluated at each time point; differentiable w.r.t. time.
    Physics: first-order saturation ODE with class-conditioned learned parameters.

    Training loss = ELBO + lambda_physics * physics_residual.
    At generation, sample z ~ N(0,I) and evaluate decoder on a standard 36-point time grid.
    """
    def __init__(self, latent_dim: int = 32, num_classes: int = 3, embed_dim: int = 16) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.T_max = T_MAX

        # ConditionalEncoder has its own internal label embedding (same as CVAE)
        self.encoder = ConditionalEncoder(latent_dim, num_classes, embed_dim)
        # Decoder and physics head share a separate label embedding
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        self.decoder = PINNDecoder(latent_dim, embed_dim)
        self.physics_head = PhysicsParamHead(embed_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = (0.5 * logvar).exp()
            return mu + std * torch.randn_like(std)
        return mu

    def _decode_at_times(self, z: torch.Tensor, y_embed: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        Evaluate decoder at a grid of time points.

        :param z: (B, latent_dim)
        :param y_embed: (B, embed_dim)
        :param time: (B, N) time in seconds
        :return: (B, 2, N) [humidity, temperature] signals
        """
        B, N = time.shape
        t_norm = (time / self.T_max).reshape(B * N, 1)
        z_flat = z.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)
        e_flat = y_embed.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)
        u = self.decoder(t_norm, z_flat, e_flat)  # (B*N, 2)
        return u.reshape(B, N, 2).permute(0, 2, 1)  # (B, 2, N)

    def _physics_residual(self, z: torch.Tensor, y_embed: torch.Tensor, n_col: int) -> torch.Tensor:
        """
        Mean-squared ODE residual at random collocation points in [0, 1] (normalized time).

        ODE:  dH/dt_norm = α*(H_sat - H) - β*H
              dT/dt_norm = γ*(T_sat - T) - δ*T

        :param z: (B, latent_dim) latent vectors
        :param y_embed: (B, embed_dim) class embeddings
        :param n_col: number of collocation points per sample
        :return: scalar residual loss
        """
        B = z.shape[0]

        # Collocation points require grad so autograd can compute du/dt
        t_col = torch.rand(B * n_col, 1, device=z.device, requires_grad=True)

        z_flat = z.unsqueeze(1).expand(B, n_col, -1).reshape(B * n_col, -1)
        e_flat = y_embed.unsqueeze(1).expand(B, n_col, -1).reshape(B * n_col, -1)

        u = self.decoder(t_col, z_flat, e_flat)  # (B*N_col, 2)
        H, T_sig = u[:, 0], u[:, 1]

        # du/dt via autograd; create_graph keeps gradients in the computation graph
        dH_dt = torch.autograd.grad(H.sum(), t_col, create_graph=True)[0].squeeze(1)   # (B*N_col,)
        dT_dt = torch.autograd.grad(T_sig.sum(), t_col, create_graph=True)[0].squeeze(1)

        # Class-conditioned physics parameters, expanded to (B*N_col,)
        alpha, beta, H_sat, gamma, delta, T_sat = self.physics_head(y_embed)
        def _expand(p):
            return p.unsqueeze(1).expand(B, n_col).reshape(B * n_col)
        alpha, beta, H_sat = _expand(alpha), _expand(beta), _expand(H_sat)
        gamma, delta, T_sat = _expand(gamma), _expand(delta), _expand(T_sat)

        res_H = dH_dt - alpha * (H_sat - H) + beta * H
        res_T = dT_dt - gamma * (T_sat - T_sig) + delta * T_sig

        return (res_H.pow(2) + res_T.pow(2)).mean()

    def forward(self, x: torch.Tensor, time: torch.Tensor, y: torch.Tensor, n_col: int = 100) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param x: (B, 2, 36) input signals (z-score normalised)
        :param time: (B, 36) time in seconds, aligned to 0
        :param y: (B,) integer class labels
        :param n_col: collocation points for physics residual (used only during training)
        :return: x_hat (B, 2, 36), mu (B, latent_dim), logvar (B, latent_dim), physics_res scalar
        """
        mu, logvar = self.encoder(x, y)
        z = self.reparameterize(mu, logvar)
        y_embed = self.label_embed(y)

        x_hat = self._decode_at_times(z, y_embed, time)

        if self.training:
            physics_res = self._physics_residual(z, y_embed, n_col)
        else:
            physics_res = torch.tensor(0.0, device=x.device)

        return x_hat, mu, logvar, physics_res

    def sample(self, n: int, y: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Sample n signals conditioned on class labels y.

        :param n: number of samples to generate
        :param y: (n,) integer class labels, or scalar broadcast to all n samples
        :param device: device to generate on
        :return: (n, 2, 36) generated signals
        """
        z = torch.randn(n, self.latent_dim, device=device)
        if y.dim() == 0:
            y = y.expand(n)
        y_embed = self.label_embed(y.to(device))

        # Standard 36-point time grid matching the dataset
        t_grid = torch.linspace(0, self.T_max, 36, device=device)
        time = t_grid.unsqueeze(0).expand(n, -1)  # (n, 36)

        self.eval()
        with torch.no_grad():
            return self._decode_at_times(z, y_embed, time)
