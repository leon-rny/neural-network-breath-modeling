import torch
import torch.nn as nn

class AblationCVAE(nn.Module):
    """Base class: subclasses assign self.encoder and self.decoder."""
    def __init__(self, latent_dim: int, num_classes: int, num_participants: int,
                 condition_on_participant: bool) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.num_participants = num_participants
        self._cond_part = condition_on_participant

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = (0.5 * logvar).exp()
            return mu + std * torch.randn_like(std)
        return mu

    def forward(self, x: torch.Tensor, y: torch.Tensor,
                p: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x, y, p)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z, y, p), mu, logvar

    def sample(self, n: int, y: torch.Tensor, device: torch.device) -> torch.Tensor:
        z = torch.randn(n, self.latent_dim, device=device)
        if y.dim() == 0:
            y = y.expand(n)
        p = torch.randint(0, self.num_participants, (n,), device=device) if self._cond_part else None
        self.eval()
        with torch.no_grad():
            return self.decoder(z, y.to(device), p)

# some helpers
def _cond_size(embed_dim: int, cond_part: bool, part_embed_dim: int) -> int:
    return embed_dim + (part_embed_dim if cond_part else 0)

def _gather_parts(h: torch.Tensor, y: torch.Tensor, p: torch.Tensor | None,
                  label_embed: nn.Embedding,
                  part_embed: nn.Embedding | None) -> torch.Tensor:
    parts = [h, label_embed(y)]
    if part_embed is not None and p is not None:
        parts.append(part_embed(p))
    return torch.cat(parts, dim=1)

# conv baseline: 3-layer Conv1d 2→16→32→64, FC 128
class _Enc_ConvBaseline(nn.Module):
    def __init__(self, latent_dim, num_classes, embed_dim, cond_part, num_participants, part_embed_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, 16, 3, padding=1), nn.ReLU(),
            nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),
        )
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        self.part_embed = nn.Embedding(num_participants, part_embed_dim) if cond_part else None
        in_fc = 64 * 36 + _cond_size(embed_dim, cond_part, part_embed_dim)
        self.fc = nn.Sequential(nn.Linear(in_fc, 128), nn.ReLU())
        self.mu_head = nn.Linear(128, latent_dim)
        self.logvar_head = nn.Linear(128, latent_dim)

    def forward(self, x, y, p=None):
        h = self.conv(x).flatten(1)
        h = self.fc(_gather_parts(h, y, p, self.label_embed, self.part_embed))
        return self.mu_head(h), self.logvar_head(h)

class _Dec_ConvBaseline(nn.Module):
    def __init__(self, latent_dim, num_classes, embed_dim, cond_part, num_participants, part_embed_dim):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        self.part_embed = nn.Embedding(num_participants, part_embed_dim) if cond_part else None
        in_fc = latent_dim + _cond_size(embed_dim, cond_part, part_embed_dim)
        self.fc = nn.Sequential(nn.Linear(in_fc, 128), nn.ReLU(), nn.Linear(128, 64 * 36))
        self.conv = nn.Sequential(
            nn.ConvTranspose1d(64, 32, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(32, 16, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(16, 2, 3, padding=1),
        )

    def forward(self, z, y, p=None):
        h = self.fc(_gather_parts(z, y, p, self.label_embed, self.part_embed)).view(-1, 64, 36)
        return self.conv(h)

class ConvBaseline(AblationCVAE):
    def __init__(self, latent_dim=16, num_classes=3, embed_dim=8,
                 condition_on_participant=False, num_participants=3, part_embed_dim=8):
        super().__init__(latent_dim, num_classes, num_participants, condition_on_participant)
        kw = dict(latent_dim=latent_dim, num_classes=num_classes, embed_dim=embed_dim,
                  cond_part=condition_on_participant, num_participants=num_participants,
                  part_embed_dim=part_embed_dim)
        self.encoder = _Enc_ConvBaseline(**kw)
        self.decoder = _Dec_ConvBaseline(**kw)

# conv_slim — 2-layer Conv1d 2→8→16, FC 64
class _Enc_ConvSlim(nn.Module):
    def __init__(self, latent_dim, num_classes, embed_dim, cond_part, num_participants, part_embed_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, 8, 3, padding=1), nn.ReLU(),
            nn.Conv1d(8, 16, 3, padding=1), nn.ReLU(),
        )
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        self.part_embed = nn.Embedding(num_participants, part_embed_dim) if cond_part else None
        in_fc = 16 * 36 + _cond_size(embed_dim, cond_part, part_embed_dim)
        self.fc = nn.Sequential(nn.Linear(in_fc, 64), nn.ReLU())
        self.mu_head = nn.Linear(64, latent_dim)
        self.logvar_head = nn.Linear(64, latent_dim)

    def forward(self, x, y, p=None):
        h = self.conv(x).flatten(1)
        h = self.fc(_gather_parts(h, y, p, self.label_embed, self.part_embed))
        return self.mu_head(h), self.logvar_head(h)

class _Dec_ConvSlim(nn.Module):
    def __init__(self, latent_dim, num_classes, embed_dim, cond_part, num_participants, part_embed_dim):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        self.part_embed = nn.Embedding(num_participants, part_embed_dim) if cond_part else None
        in_fc = latent_dim + _cond_size(embed_dim, cond_part, part_embed_dim)
        self.fc = nn.Sequential(nn.Linear(in_fc, 64), nn.ReLU(), nn.Linear(64, 16 * 36))
        self.conv = nn.Sequential(
            nn.ConvTranspose1d(16, 8, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(8, 2, 3, padding=1),
        )

    def forward(self, z, y, p=None):
        h = self.fc(_gather_parts(z, y, p, self.label_embed, self.part_embed)).view(-1, 16, 36)
        return self.conv(h)

class ConvSlim(AblationCVAE):
    def __init__(self, latent_dim=16, num_classes=3, embed_dim=8,
                 condition_on_participant=False, num_participants=3, part_embed_dim=8):
        super().__init__(latent_dim, num_classes, num_participants, condition_on_participant)
        kw = dict(latent_dim=latent_dim, num_classes=num_classes, embed_dim=embed_dim,
                  cond_part=condition_on_participant, num_participants=num_participants,
                  part_embed_dim=part_embed_dim)
        self.encoder = _Enc_ConvSlim(**kw)
        self.decoder = _Dec_ConvSlim(**kw)

# mlp — no convolutions, 72→128→64→μ/σ
class _Enc_MLP(nn.Module):
    def __init__(self, latent_dim, num_classes, embed_dim, cond_part, num_participants, part_embed_dim):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        self.part_embed = nn.Embedding(num_participants, part_embed_dim) if cond_part else None
        in_fc = 72 + _cond_size(embed_dim, cond_part, part_embed_dim)
        self.fc = nn.Sequential(nn.Linear(in_fc, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.mu_head = nn.Linear(64, latent_dim)
        self.logvar_head = nn.Linear(64, latent_dim)

    def forward(self, x, y, p=None):
        h = x.flatten(1)  # (B, 72)
        h = self.fc(_gather_parts(h, y, p, self.label_embed, self.part_embed))
        return self.mu_head(h), self.logvar_head(h)

class _Dec_MLP(nn.Module):
    def __init__(self, latent_dim, num_classes, embed_dim, cond_part, num_participants, part_embed_dim):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        self.part_embed = nn.Embedding(num_participants, part_embed_dim) if cond_part else None
        in_fc = latent_dim + _cond_size(embed_dim, cond_part, part_embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(in_fc, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 72),
        )

    def forward(self, z, y, p=None):
        return self.fc(_gather_parts(z, y, p, self.label_embed, self.part_embed)).view(-1, 2, 36)

class MLP(AblationCVAE):
    def __init__(self, latent_dim=16, num_classes=3, embed_dim=8,
                 condition_on_participant=False, num_participants=3, part_embed_dim=8):
        super().__init__(latent_dim, num_classes, num_participants, condition_on_participant)
        kw = dict(latent_dim=latent_dim, num_classes=num_classes, embed_dim=embed_dim,
                  cond_part=condition_on_participant, num_participants=num_participants,
                  part_embed_dim=part_embed_dim)
        self.encoder = _Enc_MLP(**kw)
        self.decoder = _Dec_MLP(**kw)

# mlp_small — 72→64→32→μ/σ
class _Enc_MLPSmall(nn.Module):
    def __init__(self, latent_dim, num_classes, embed_dim, cond_part, num_participants, part_embed_dim):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        self.part_embed = nn.Embedding(num_participants, part_embed_dim) if cond_part else None
        in_fc = 72 + _cond_size(embed_dim, cond_part, part_embed_dim)
        self.fc = nn.Sequential(nn.Linear(in_fc, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.mu_head = nn.Linear(32, latent_dim)
        self.logvar_head = nn.Linear(32, latent_dim)

    def forward(self, x, y, p=None):
        h = x.flatten(1)
        h = self.fc(_gather_parts(h, y, p, self.label_embed, self.part_embed))
        return self.mu_head(h), self.logvar_head(h)

class _Dec_MLPSmall(nn.Module):
    def __init__(self, latent_dim, num_classes, embed_dim, cond_part, num_participants, part_embed_dim):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        self.part_embed = nn.Embedding(num_participants, part_embed_dim) if cond_part else None
        in_fc = latent_dim + _cond_size(embed_dim, cond_part, part_embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(in_fc, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 72),
        )

    def forward(self, z, y, p=None):
        return self.fc(_gather_parts(z, y, p, self.label_embed, self.part_embed)).view(-1, 2, 36)

class MLPSmall(AblationCVAE):
    def __init__(self, latent_dim=16, num_classes=3, embed_dim=8,
                 condition_on_participant=False, num_participants=3, part_embed_dim=8):
        super().__init__(latent_dim, num_classes, num_participants, condition_on_participant)
        kw = dict(latent_dim=latent_dim, num_classes=num_classes, embed_dim=embed_dim,
                  cond_part=condition_on_participant, num_participants=num_participants,
                  part_embed_dim=part_embed_dim)
        self.encoder = _Enc_MLPSmall(**kw)
        self.decoder = _Dec_MLPSmall(**kw)

# conv_asym — full conv encoder, single-layer decoder with dropout
class _Dec_ConvAsym(nn.Module):
    def __init__(self, latent_dim, num_classes, embed_dim, cond_part, num_participants, part_embed_dim):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        self.part_embed = nn.Embedding(num_participants, part_embed_dim) if cond_part else None
        in_fc = latent_dim + _cond_size(embed_dim, cond_part, part_embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(in_fc, 128), nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64 * 36),
        )
        self.conv = nn.ConvTranspose1d(64, 2, 3, padding=1)  # single deconv layer

    def forward(self, z, y, p=None):
        h = self.fc(_gather_parts(z, y, p, self.label_embed, self.part_embed)).view(-1, 64, 36)
        return self.conv(h)

class ConvAsym(AblationCVAE):
    """Baseline encoder + weak single-layer decoder with dropout — forces z to carry signal."""

    def __init__(self, latent_dim=16, num_classes=3, embed_dim=8,
                 condition_on_participant=False, num_participants=3, part_embed_dim=8):
        super().__init__(latent_dim, num_classes, num_participants, condition_on_participant)
        kw = dict(latent_dim=latent_dim, num_classes=num_classes, embed_dim=embed_dim,
                  cond_part=condition_on_participant, num_participants=num_participants,
                  part_embed_dim=part_embed_dim)
        self.encoder = _Enc_ConvBaseline(**kw)
        self.decoder = _Dec_ConvAsym(**kw)

VARIANT_MAP: dict[str, type] = {"conv_baseline": ConvBaseline,
                                "conv_slim": ConvSlim,
                                "mlp": MLP,
                                "mlp_small": MLPSmall,
                                "conv_asym": ConvAsym,}
