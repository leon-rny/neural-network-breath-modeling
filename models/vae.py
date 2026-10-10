"""Conditional VAE (class and participant conditioning, participant dropout to a null token) and its ELBO loss."""

import torch
from torch import nn


def _expand_embedding_with_null(emb: nn.Embedding) -> nn.Embedding:
    # copy of emb with one extra (null-token) row; the caller saves / restores the global RNG so that the surrounding
    # initialisations stay unperturbed
    n, d = emb.weight.shape
    new = nn.Embedding(n + 1, d)
    with torch.no_grad():
        new.weight[:n] = emb.weight
    return new

def elbo_loss(x: torch.Tensor, x_hat: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0, free_bits: float = 0.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MSE reconstruction + beta x KL divergence.

    :param beta: weight of the KL term (warmup)
    :param free_bits: minimum KL per latent dimension
    :return: (loss, reconstruction, KL)
    """
    recon = nn.functional.mse_loss(x_hat, x, reduction='mean')
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (B, latent_dim)
    kl_per_dim = kl_per_dim.mean(dim=0)  # (latent_dim,)
    kl = torch.clamp(kl_per_dim, min=free_bits).mean()
    return recon + beta * kl, recon, kl

# cvae
class ConditionalEncoder(nn.Module):
    """Conv encoder; class (and participant) embeddings are concatenated to the flattened conv features."""

    def __init__(self, latent_dim: int = 16, num_classes: int = 3, embed_dim: int = 8, condition_on_participant: bool = False, num_participants: int = 3, part_embed_dim: int = 8) -> None:
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
        h = self.conv(x).flatten(1)
        parts = [h, self.label_embed(y)]
        if self._cond_part:
            parts.append(self.part_embed(p))
        h = self.fc(torch.cat(parts, dim=1))
        return self.mu_head(h), self.logvar_head(h)

class ConditionalDecoder(nn.Module):
    """Conv decoder; the latent vector is concatenated with the class (and participant) embeddings."""

    def __init__(self, latent_dim: int = 16, num_classes: int = 3, embed_dim: int = 8, condition_on_participant: bool = False, num_participants: int = 3, part_embed_dim: int = 8) -> None:
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
        parts = [z, self.label_embed(y)]
        if self._cond_part:
            parts.append(self.part_embed(p))
        h = self.fc(torch.cat(parts, dim=1))
        h = h.view(h.size(0), 64, 36)
        return self.conv(h)

class CVAE(nn.Module):
    """Conditional VAE on class and optionally participant. With part_dropout > 0 the participant is replaced by a null
    token during training; an unseen participant (LOSO) is decoded with that token (null_part_idx)."""

    def __init__(self, latent_dim: int = 16, num_classes: int = 3, embed_dim: int = 8, condition_on_participant: bool = False, num_participants: int = 3, part_embed_dim: int = 8, part_dropout: float = 0.0) -> None:
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
        # z ~ N(mu, sigma) while training, mu at evaluation
        if self.training:
            std = (0.5 * logvar).exp()
            return mu + std * torch.randn_like(std)
        return mu

    def forward(self, x: torch.Tensor, y: torch.Tensor, p: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode and decode (B, 2, 36) signals with class labels y and participant labels p.

        :return: (reconstruction, mu, logvar)
        """
        # training-only participant dropout
        if self.training and self._cond_part and p is not None and self.part_dropout > 0.0 and torch.rand(1).item() < self.part_dropout:
            p = torch.full_like(p, self.null_part_idx)
        # participants the model was not trained on -> null token
        if not self.training and self._cond_part and p is not None and hasattr(self, 'trained_participants'):
            known = torch.tensor(self.trained_participants, device=p.device)
            p = torch.where(torch.isin(p, known), p, self.null_part_idx)
        mu, logvar = self.encoder(x, y, p)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, y, p)
        return x_hat, mu, logvar

    def decode(self, z: torch.Tensor, y: torch.Tensor, p: torch.Tensor | None = None) -> torch.Tensor:
        """Latent vectors + class (and participant) labels -> (B, 2, 36) signals in the training space."""
        return self.decoder(z, y, p)
