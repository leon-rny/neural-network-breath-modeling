import torch
import torch.nn as nn

def _pe(p_embed: torch.Tensor, batch: int) -> torch.Tensor:
    """Broadcast a (D,) or (B, D) participant embedding to (batch, D) for concatenation (few-shot adaptation)."""
    return p_embed.unsqueeze(0).expand(batch, -1) if p_embed.dim() == 1 else p_embed

def _expand_embedding_with_null(emb: nn.Embedding) -> nn.Embedding:
    """Return a copy of `emb` with one extra (null-token) row appended.
    Existing rows are copied byte-for-byte, only the new row is freshly initialised.
    The caller must save/restore the global RNG around this so surrounding inits stay unperturbed."""
    n, d = emb.weight.shape
    new = nn.Embedding(n + 1, d)
    with torch.no_grad():
        new.weight[:n] = emb.weight
    return new

def soft_dtw(X: torch.Tensor, Y: torch.Tensor, gamma: float = 0.1) -> torch.Tensor:
    """Batched differentiable soft-DTW (Cuturi & Blondel 2017) between (B, C, T) sequences — elastic-alignment
    reconstruction loss that tolerates onset/phase jitter (research idea #3, proper). Returns (B,) distances / T.

    Vectorized over anti-diagonals: all cells (i,j) with i+j=k are independent given diagonals k-1,k-2, so the DP
    is 2T-1 Python steps (not T^2), each a batched tensor op. A cell indexed by its row i on a rolling (B,T+1)
    diagonal; the three DTW predecessors become same-index / shifted-by-one reads of the two prior diagonals."""
    B, C, T = X.shape
    Xt, Yt = X.transpose(1, 2), Y.transpose(1, 2)  # (B,T,C)
    D = ((Xt.unsqueeze(2) - Yt.unsqueeze(1)) ** 2).sum(-1)  # (B,T,T) squared-euclid per (i,j)
    INF = 1e10
    rows = torch.arange(1, T + 1, device=X.device)  # cell rows i=1..T (index 1..T of the (B,T+1) diagonal)
    d_prev2 = X.new_full((B, T + 1), INF)            # diagonal k-2 (indexed by row i, col 0 = border)
    d_prev1 = X.new_full((B, T + 1), INF)            # diagonal k-1
    d_prev2[:, 0] = 0.0                              # R[0][0] = 0 (only finite border cell)
    for k in range(2, 2 * T + 1):                    # anti-diagonals; cell (i,j=k-i)
        j = k - rows                                 # (T,) column per row
        valid = (j >= 1) & (j <= T)                  # in-bounds cells on this diagonal
        jc = j.clamp(1, T)
        cost = D[:, rows - 1, jc - 1]                # (B,T) local cost for each row's cell
        b = d_prev1[:, 1:]                           # R[i][j-1]  (same row i on diag k-1)
        a = d_prev1[:, 0:T]                          # R[i-1][j]  (row i-1 on diag k-1)
        c = d_prev2[:, 0:T]                          # R[i-1][j-1](row i-1 on diag k-2)
        m = torch.min(torch.min(a, b), c)
        softmin = -gamma * torch.log(torch.exp(-(a - m) / gamma) + torch.exp(-(b - m) / gamma) + torch.exp(-(c - m) / gamma)) + m
        cur = cost + softmin
        cur = torch.where(valid.unsqueeze(0).expand(B, -1), cur, X.new_full((1,), INF))
        d_cur = torch.cat([X.new_full((B, 1), INF), cur], dim=1)  # prepend col-0 border
        d_prev2, d_prev1 = d_prev1, d_cur
    return d_prev1[:, T] / T                         # R[T][T] sits at row T of the last diagonal

def mmd_rbf(x: torch.Tensor, y: torch.Tensor, sigmas=(1.0, 2.0, 4.0, 8.0, 16.0)) -> torch.Tensor:
    """Multi-bandwidth RBF MMD^2 between two sample sets (B, D) — used for InfoVAE-style latent matching
    MMD(q(z), p(z)) to close the prior-posterior gap that causes poor generative coverage (research idea #6)."""
    def k(a, b):
        d = torch.cdist(a, b) ** 2
        return sum(torch.exp(-d / (2 * s * s)) for s in sigmas)
    return k(x, x).mean() + k(y, y).mean() - 2 * k(x, y).mean()

def elbo_loss(x: torch.Tensor, x_hat: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0, free_bits: float = 0.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ELBO loss = MSE reconstruction + beta * KL divergence.

    :param x: (B, 2, 36) original signals
    :param x_hat: (B, 2, 36) reconstructed signals
    :param mu: (B, latent_dim) mean of latent distribution
    :param logvar: (B, latent_dim) log-variance of latent distribution
    :param beta: weight for KL divergence (for annealing)
    :param free_bits: minimum KL per dimension to prevent posterior collapse
    :return: total loss, reconstruction loss, KL divergence
    """
    recon = nn.functional.mse_loss(x_hat, x, reduction='mean')
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (B, latent_dim)
    kl_per_dim = kl_per_dim.mean(dim=0) # (latent_dim,)
    kl = torch.clamp(kl_per_dim, min=free_bits).mean()
    return recon + beta * kl, recon, kl

# vae
class Encoder(nn.Module):
    """Compress a raw signal (B, 2, 36) into a compact description of its distribution in latent space.
    Convolutional layers: first layer extracts simple features, later layers combine those into more complex patterns.
    Fully connected bottleneck: maps the extracted features into a latent distribution (mu, logvar).
    """
    def __init__(self, latent_dim: int = 16) -> None:
        """
        :param latent_dim: Dimension of the latent space (default 16).
        """
        super().__init__()
        # define convolutional layers to extract features from the input signal
        self.conv = nn.Sequential(nn.Conv1d(2, 16, kernel_size=3, padding=1),
                                  nn.ReLU(),
                                  nn.Conv1d(16, 32, kernel_size=3, padding=1),
                                  nn.ReLU(),
                                  nn.Conv1d(32, 64, kernel_size=3, padding=1),
                                  nn.ReLU())

        # define fully connected bottleneck
        self.fc = nn.Sequential(nn.Linear(64*36, 128),
                                nn.ReLU())
        self.mu_head = nn.Linear(128, latent_dim)
        self.logvar_head = nn.Linear(128, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Defines forward pass through encoder

        :param x: (B, 2, 36) input signals
        :return: (B, latent_dim) mean and log-variance of latent distribution
        """
        h = self.conv(x) # (B, 64, 36)
        h = h.flatten(1) # (B, 64*36)
        h = self.fc(h) # (B, 128)
        return self.mu_head(h), self.logvar_head(h)

class Decoder(nn.Module):
    """Mirror of the encoder that maps a latent vector back to the original signal space.
    """
    def __init__(self, latent_dim: int = 16) -> None:
        """
        :param latent_dim: Dimension of the latent space (default 16).
        """
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(latent_dim, 128),
                                nn.ReLU(),
                                nn.Linear(128, 64 * 36))

        # use transposed convolutions to "deconvolve"
        self.conv = nn.Sequential(nn.ConvTranspose1d(64, 32, kernel_size=3, padding=1),
                                  nn.ReLU(),
                                  nn.ConvTranspose1d(32, 16, kernel_size=3, padding=1),
                                  nn.ReLU(),
                                  nn.ConvTranspose1d(16, 2, kernel_size=3, padding=1)) # no activation on final layer

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Defines forward pass through decoder

        :param z: (B, latent_dim) latent vectors
        :return: (B, 2, 36) reconstructed signals
        """
        h = self.fc(z) # (B, 64*36)
        h = h.view(h.size(0), 64, 36) # (B, 64, 36)
        return self.conv(h) # (B, 2, 36)

class VAE(nn.Module):
    """Combines the encoder and decoder, implements the reparameterization trick and defines a sampling method."""
    def __init__(self, latent_dim: int = 16) -> None:
        """
        :param latent_dim: Dimension of the latent space (default 16).
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: during training, sample from the latent distribution using mu and logvar.

        :param mu: (B, latent_dim) mean of the latent distribution
        :param logvar: (B, latent_dim) log-variance of the latent distribution
        :return: (B, latent_dim) sampled latent vector
        """
        if self.training:
            std = (0.5 * logvar).exp()
            return mu + std * torch.randn_like(std)
        return mu

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the VAE: encode input, sample from latent distribution, decode to reconstruct.

        :param x: (B, 2, 36) input signals
        :return: (B, 2, 36) reconstructed signals, (B, latent_dim) mean, (B, latent_dim) log-variance
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        """Sample n signals from the prior N(0, I).

        :param n: number of samples to generate
        :param device: device to perform sampling on
        :return: (n, 2, 36) generated signals
        """
        z = torch.randn(n, self.latent_dim, device=device)
        self.eval()
        with torch.no_grad():
            return self.decoder(z)

# cvae
class ConditionalEncoder(nn.Module):
    """Conditional variant of Encoder: conditions the latent distribution on a class label.
    Optionally also conditions on a participant label (condition_on_participant=True).
    The embeddings are concatenated to the flattened conv features before the FC bottleneck.
    """
    def __init__(self, latent_dim: int = 16, num_classes: int = 3, embed_dim: int = 8, condition_on_participant: bool = False, num_participants: int = 3, part_embed_dim: int = 8) -> None:
        """
        :param latent_dim: Dimension of the latent space (default 16).
        :param num_classes: Number of breath classes to condition on (default 3).
        :param embed_dim: Dimension of the class embedding (default 8).
        :param condition_on_participant: Whether to also condition on participant id (default False).
        :param num_participants: Number of participants (default 3).
        :param part_embed_dim: Dimension of the participant embedding (default 8).
        """
        super().__init__()
        self._cond_part = condition_on_participant
        self.conv = nn.Sequential(nn.Conv1d(2, 16, kernel_size=3, padding=1),
                                  nn.ReLU(),
                                  nn.Conv1d(16, 32, kernel_size=3, padding=1),
                                  nn.ReLU(),
                                  nn.Conv1d(32, 64, kernel_size=3, padding=1),
                                  nn.ReLU())
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        extra = 0
        if condition_on_participant:
            self.part_embed = nn.Embedding(num_participants, part_embed_dim)
            extra = part_embed_dim
        self.fc = nn.Sequential(nn.Linear(64 * 36 + embed_dim + extra, 128), nn.ReLU())
        self.mu_head = nn.Linear(128, latent_dim)
        self.logvar_head = nn.Linear(128, latent_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor, p: torch.Tensor | None = None, p_embed: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: (B, 2, 36) input signals
        :param y: (B,) integer class labels
        :param p: (B,) integer participant labels (required when condition_on_participant=True)
        :param p_embed: optional (part_embed_dim,) or (B, part_embed_dim) participant embedding used INSTEAD of
            the learned lookup part_embed(p) (few-shot subject adaptation)
        :return: (B, latent_dim) mean and log-variance
        """
        h = self.conv(x).flatten(1)
        parts = [h, self.label_embed(y)]
        if self._cond_part:
            parts.append(_pe(p_embed, h.size(0)) if p_embed is not None else self.part_embed(p))
        h = self.fc(torch.cat(parts, dim=1))
        return self.mu_head(h), self.logvar_head(h)

class ConditionalDecoder(nn.Module):
    """Conditional variant of Decoder: the latent vector is concatenated with a class embedding before reconstruction.
    Optionally also conditions on a participant label (condition_on_participant=True).
    """
    def __init__(self, latent_dim: int = 16, num_classes: int = 3, embed_dim: int = 8, condition_on_participant: bool = False, num_participants: int = 3, part_embed_dim: int = 8) -> None:
        """
        :param latent_dim: Dimension of the latent space (default 16).
        :param num_classes: Number of breath classes to condition on (default 3).
        :param embed_dim: Dimension of the class embedding (default 8).
        :param condition_on_participant: Whether to also condition on participant id (default False).
        :param num_participants: Number of participants (default 3).
        :param part_embed_dim: Dimension of the participant embedding (default 8).
        """
        super().__init__()
        self._cond_part = condition_on_participant
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        extra = 0
        if condition_on_participant:
            self.part_embed = nn.Embedding(num_participants, part_embed_dim)
            extra = part_embed_dim
        self.fc = nn.Sequential(nn.Linear(latent_dim + embed_dim + extra, 128),
                                nn.ReLU(),
                                nn.Linear(128, 64 * 36))
        self.conv = nn.Sequential(nn.ConvTranspose1d(64, 32, kernel_size=3, padding=1),
                                  nn.ReLU(),
                                  nn.ConvTranspose1d(32, 16, kernel_size=3, padding=1),
                                  nn.ReLU(),
                                  nn.ConvTranspose1d(16, 2, kernel_size=3, padding=1))

    def forward(self, z: torch.Tensor, y: torch.Tensor, p: torch.Tensor | None = None, p_embed: torch.Tensor | None = None) -> torch.Tensor:
        """
        :param z: (B, latent_dim) latent vectors
        :param y: (B,) integer class labels
        :param p: (B,) integer participant labels (required when condition_on_participant=True)
        :param p_embed: optional (part_embed_dim,) or (B, part_embed_dim) participant embedding used INSTEAD of
            the learned lookup part_embed(p) (few-shot subject adaptation)
        :return: (B, 2, 36) reconstructed signals
        """
        parts = [z, self.label_embed(y)]
        if self._cond_part:
            parts.append(_pe(p_embed, z.size(0)) if p_embed is not None else self.part_embed(p))
        h = self.fc(torch.cat(parts, dim=1))
        h = h.view(h.size(0), 64, 36)
        return self.conv(h)

class _GradReverse(torch.autograd.Function):
    """Gradient Reversal Layer: identity forward, negated (*lambda) gradient backward (DANN)."""
    @staticmethod
    def forward(ctx, x, lambd):
        """Identity forward pass; stash lambda for the backward negation."""
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """Negate and scale the incoming gradient by lambda (gradient reversal)."""
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd: float):
    """Apply the gradient reversal layer to `x` with reversal strength `lambd`."""
    return _GradReverse.apply(x, lambd)

class CVAE(nn.Module):
    """Conditional VAE conditioning on class label.
    Set condition_on_participant=True to also condition on participant during training;
    at generation time participant is sampled uniformly to marginalise over identity.
    """
    def __init__(self, latent_dim: int = 16, num_classes: int = 3, embed_dim: int = 8, condition_on_participant: bool = False, num_participants: int = 3, part_embed_dim: int = 8, part_dropout: float = 0.0, subj_adv: bool = False) -> None:
        """
        :param latent_dim: Dimension of the latent space (default 16).
        :param num_classes: Number of breath classes to condition on (default 3).
        :param embed_dim: Dimension of the class embedding (default 8).
        :param condition_on_participant: Whether to also condition on participant id (default False).
        :param num_participants: Number of participants (default 3).
        :param part_embed_dim: Dimension of the participant embedding (default 8).
        :param part_dropout: Probability of dropping participant conditioning to the null token during training (default 0.0).
        :param subj_adv: Whether to add a gradient-reversal subject-adversary head on the latent (default False).
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.num_participants = num_participants
        self._cond_part = condition_on_participant
        # participant dropout
        self.part_dropout = part_dropout
        self.null_part_idx = num_participants
        self._null_steps = 0
        self._total_steps = 0
        self.encoder = ConditionalEncoder(latent_dim, num_classes, embed_dim, condition_on_participant, num_participants, part_embed_dim)
        self.decoder = ConditionalDecoder(latent_dim, num_classes, embed_dim, condition_on_participant, num_participants, part_embed_dim)
        if condition_on_participant and part_dropout > 0.0:
            rng_state = torch.get_rng_state()
            self.encoder.part_embed = _expand_embedding_with_null(self.encoder.part_embed)
            self.decoder.part_embed = _expand_embedding_with_null(self.decoder.part_embed)
            torch.set_rng_state(rng_state)
        self.subj_adv = subj_adv
        if subj_adv:
            self.subj_clf = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, num_participants))

    def adv_logits(self, z: torch.Tensor, lambd: float) -> torch.Tensor:
        """Subject-classifier logits from z through the gradient reversal layer (DANN adversary)."""
        return self.subj_clf(grad_reverse(z, lambd))

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: sample z ~ N(mu, sigma) while training, return mu at eval."""
        if self.training:
            std = (0.5 * logvar).exp()
            return mu + std * torch.randn_like(std)
        return mu

    def null_fire_frac(self) -> float:
        """Fraction of training steps that fired the participant null token (dropout monitor)."""
        return self._null_steps / self._total_steps if self._total_steps else 0.0

    def forward(self, x: torch.Tensor, y: torch.Tensor, p: torch.Tensor | None = None, p_embed_enc: torch.Tensor | None = None, p_embed_dec: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param x: (B, 2, 36) input signals
        :param y: (B,) integer class labels
        :param p: (B,) integer participant labels (required when condition_on_participant=True)
        :param p_embed_enc, p_embed_dec: optional fitted participant embeddings (few-shot) overriding the
            encoder/decoder lookup respectively.
        :return: (B, 2, 36) reconstructed signals, (B, latent_dim) mu, (B, latent_dim) logvar
        """
        # training-only participant dropout
        if self.training and self._cond_part and p is not None and self.part_dropout > 0.0:
            self._total_steps += 1
            if torch.rand(1).item() < self.part_dropout:
                p = torch.full_like(p, self.null_part_idx)
                self._null_steps += 1
        mu, logvar = self.encoder(x, y, p, p_embed=p_embed_enc)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z, y, p, p_embed=p_embed_dec)
        return x_hat, mu, logvar

    def sample_embed(self, n: int, y: torch.Tensor, device: torch.device, p_embed_dec: torch.Tensor) -> torch.Tensor:
        """Sample n signals conditioned on class y, using a GIVEN decoder participant embedding
        (few-shot subject adaptation) instead of a learned lookup / null token.

        :param p_embed_dec: (part_embed_dim,) fitted decoder participant embedding.
        :return: (n, 2, 36) generated signals.
        """
        z = torch.randn(n, self.latent_dim, device=device)
        if y.dim() == 0:
            y = y.expand(n)
        self.eval()
        with torch.no_grad():
            return self.decoder(z, y.to(device), None, p_embed=p_embed_dec.to(device))

    def sample(self, n: int, y: torch.Tensor, device: torch.device, participant: int | None = None) -> torch.Tensor:
        """Sample n signals conditioned on class labels y.
        When condition_on_participant=True, participant is sampled uniformly over real subjects,
        unless `participant` is given (e.g. null_part_idx for LOSO generation of an unseen subject).

        :param n: number of samples to generate
        :param y: (n,) integer class labels (or scalar broadcast to all n samples)
        :param device: device to perform sampling on
        :param participant: fixed participant id to condition on (overrides uniform sampling)
        :return: (n, 2, 36) generated signals
        """
        z = torch.randn(n, self.latent_dim, device=device)
        if y.dim() == 0:
            y = y.expand(n)
        if not self._cond_part:
            p = None
        elif participant is not None:
            p = torch.full((n,), participant, dtype=torch.long, device=device)
        else:
            p = torch.randint(0, self.num_participants, (n,), device=device)
        self.eval()
        with torch.no_grad():
            return self.decoder(z, y.to(device), p)
