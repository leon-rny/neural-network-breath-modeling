import torch
from torch import nn


def _expand_embedding_with_null(emb: nn.Embedding) -> nn.Embedding:
    """Return a copy of `emb` with one extra (null-token) row appended.
    Existing rows are copied byte-for-byte, only the new row is freshly initialised.
    The caller must save/restore the global RNG around this so surrounding inits stay unperturbed."""
    n, d = emb.weight.shape
    new = nn.Embedding(n + 1, d)
    with torch.no_grad():
        new.weight[:n] = emb.weight
    return new

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
    kl_per_dim = kl_per_dim.mean(dim=0)  # (latent_dim,)
    kl = torch.clamp(kl_per_dim, min=free_bits).mean()
    return recon + beta * kl, recon, kl

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

    def forward(self, x: torch.Tensor, y: torch.Tensor, p: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: (B, 2, 36) input signals
        :param y: (B,) integer class labels
        :param p: (B,) integer participant labels (required when condition_on_participant=True)
        :return: (B, latent_dim) mean and log-variance
        """
        h = self.conv(x).flatten(1)
        parts = [h, self.label_embed(y)]
        if self._cond_part:
            parts.append(self.part_embed(p))
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

    def forward(self, z: torch.Tensor, y: torch.Tensor, p: torch.Tensor | None = None) -> torch.Tensor:
        """
        :param z: (B, latent_dim) latent vectors
        :param y: (B,) integer class labels
        :param p: (B,) integer participant labels (required when condition_on_participant=True)
        :return: (B, 2, 36) reconstructed signals
        """
        parts = [z, self.label_embed(y)]
        if self._cond_part:
            parts.append(self.part_embed(p))
        h = self.fc(torch.cat(parts, dim=1))
        h = h.view(h.size(0), 64, 36)
        return self.conv(h)

class CVAE(nn.Module):
    """Conditional VAE conditioning on class label and optionally on participant (condition_on_participant=True).
    With part_dropout > 0 the participant is replaced by a null token during training; generation for an unseen
    participant (LOSO) decodes with that token (null_part_idx).
    """
    def __init__(self, latent_dim: int = 16, num_classes: int = 3, embed_dim: int = 8, condition_on_participant: bool = False, num_participants: int = 3, part_embed_dim: int = 8, part_dropout: float = 0.0) -> None:
        """
        :param latent_dim: Dimension of the latent space (default 16).
        :param num_classes: Number of breath classes to condition on (default 3).
        :param embed_dim: Dimension of the class embedding (default 8).
        :param condition_on_participant: Whether to also condition on participant id (default False).
        :param num_participants: Number of participants (default 3).
        :param part_embed_dim: Dimension of the participant embedding (default 8).
        :param part_dropout: Probability of dropping participant conditioning to the null token during training (default 0.0).
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.num_participants = num_participants
        self._cond_part = condition_on_participant
        # participant dropout
        self.part_dropout = part_dropout
        self.null_part_idx = num_participants
        self.encoder = ConditionalEncoder(latent_dim, num_classes, embed_dim, condition_on_participant, num_participants, part_embed_dim)
        self.decoder = ConditionalDecoder(latent_dim, num_classes, embed_dim, condition_on_participant, num_participants, part_embed_dim)
        if condition_on_participant and part_dropout > 0.0:
            rng_state = torch.get_rng_state()
            self.encoder.part_embed = _expand_embedding_with_null(self.encoder.part_embed)
            self.decoder.part_embed = _expand_embedding_with_null(self.decoder.part_embed)
            torch.set_rng_state(rng_state)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: sample z ~ N(mu, sigma) while training, return mu at eval."""
        if self.training:
            std = (0.5 * logvar).exp()
            return mu + std * torch.randn_like(std)
        return mu

    def forward(self, x: torch.Tensor, y: torch.Tensor, p: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param x: (B, 2, 36) input signals
        :param y: (B,) integer class labels
        :param p: (B,) integer participant labels (required when condition_on_participant=True)
        :return: (B, 2, 36) reconstructed signals, (B, latent_dim) mu, (B, latent_dim) logvar
        """
        # training-only participant dropout
        if self.training and self._cond_part and p is not None and self.part_dropout > 0.0:
            if torch.rand(1).item() < self.part_dropout:
                p = torch.full_like(p, self.null_part_idx)
        if not self.training and self._cond_part and p is not None and hasattr(self, 'trained_participants'):
            known = torch.tensor(self.trained_participants, device=p.device)
            p = torch.where(torch.isin(p, known), p, self.null_part_idx)
        mu, logvar = self.encoder(x, y, p)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, y, p)
        return x_hat, mu, logvar

    def decode(self, z: torch.Tensor, y: torch.Tensor, p: torch.Tensor | None = None) -> torch.Tensor:
        """Decode latent vectors with class (and participant) labels into (B, 2, 36) signals in the training space."""
        return self.decoder(z, y, p)
