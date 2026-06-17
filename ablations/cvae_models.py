import torch
import torch.nn as nn

def _expand_embedding_with_null(emb: nn.Embedding) -> nn.Embedding:
    """Return a copy of `emb` with one extra (null-token) row appended.
    Existing rows are copied byte-for-byte; only the new row is freshly initialised.
    The caller must save/restore the global RNG around this so surrounding inits stay unperturbed."""
    n, d = emb.weight.shape
    new = nn.Embedding(n + 1, d)
    with torch.no_grad():
        new.weight[:n] = emb.weight
    return new

class AblationCVAE(nn.Module):
    def __init__(self, latent_dim: int, num_classes: int, num_participants: int, condition_on_participant: bool, part_dropout: float = 0.0) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.num_participants = num_participants
        self._cond_part = condition_on_participant
        # CFG-style participant dropout: with prob part_dropout swap the batch to a learned null token.
        # null row sits just past the real participants; only allocated when part_dropout > 0 (see subclass).
        self.part_dropout = part_dropout
        self.null_part_idx = num_participants
        self._null_steps = 0
        self._total_steps = 0

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = (0.5 * logvar).exp()
            return mu + std * torch.randn_like(std)
        return mu

    def null_fire_frac(self) -> float:
        return self._null_steps / self._total_steps if self._total_steps else 0.0

    def forward(self, x: torch.Tensor, y: torch.Tensor, p: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # training-only participant dropout; guarded so part_dropout==0 draws no RNG (run A stays byte-identical)
        if self.training and self._cond_part and p is not None and self.part_dropout > 0.0:
            self._total_steps += 1
            if torch.rand(1).item() < self.part_dropout:
                p = torch.full_like(p, self.null_part_idx)
                self._null_steps += 1
        mu, logvar = self.encoder(x, y, p)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z, y, p), mu, logvar

    def sample(self, n: int, y: torch.Tensor, device: torch.device, participant: int | None = None) -> torch.Tensor:
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
            return self.decoder(z, y.to(device), p)

# helpers
def _cond_size(embed_dim: int, cond_part: bool, part_embed_dim: int) -> int:
    return embed_dim + (part_embed_dim if cond_part else 0)

def _gather_parts(h: torch.Tensor, y: torch.Tensor, p: torch.Tensor | None, label_embed: nn.Embedding, part_embed: nn.Embedding | None) -> torch.Tensor:
    parts = [h, label_embed(y)]
    if part_embed is not None and p is not None:
        parts.append(part_embed(p))
    return torch.cat(parts, dim=1)

# conv baseline: 3-layer Conv1d 2->16->32->64, FC 128
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
    def __init__(self, latent_dim=16, num_classes=3, embed_dim=8, condition_on_participant=False, num_participants=3, part_embed_dim=8, part_dropout=0.0):
        super().__init__(latent_dim, num_classes, num_participants, condition_on_participant, part_dropout)
        kw = dict(latent_dim=latent_dim, num_classes=num_classes, embed_dim=embed_dim, cond_part=condition_on_participant, num_participants=num_participants, part_embed_dim=part_embed_dim)
        self.encoder = _Enc_ConvBaseline(**kw)
        self.decoder = _Dec_ConvBaseline(**kw)
        # append the null-token row last, with RNG save/restore so every other param keeps its exact draw
        # (part_dropout==0 → no expansion → byte-identical to the committed model)
        if condition_on_participant and part_dropout > 0.0:
            rng_state = torch.get_rng_state()
            self.encoder.part_embed = _expand_embedding_with_null(self.encoder.part_embed)
            self.decoder.part_embed = _expand_embedding_with_null(self.decoder.part_embed)
            torch.set_rng_state(rng_state)

# conv_large_kernel: same shape as ConvBaseline but kernel=7 throughout
class _Enc_ConvLargeKernel(nn.Module):
    def __init__(self, latent_dim, num_classes, embed_dim, cond_part, num_participants, part_embed_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, 16, 7, padding=3), nn.ReLU(),
            nn.Conv1d(16, 32, 7, padding=3), nn.ReLU(),
            nn.Conv1d(32, 64, 7, padding=3), nn.ReLU(),
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

class _Dec_ConvLargeKernel(nn.Module):
    def __init__(self, latent_dim, num_classes, embed_dim, cond_part, num_participants, part_embed_dim):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        self.part_embed = nn.Embedding(num_participants, part_embed_dim) if cond_part else None
        in_fc = latent_dim + _cond_size(embed_dim, cond_part, part_embed_dim)
        self.fc = nn.Sequential(nn.Linear(in_fc, 128), nn.ReLU(), nn.Linear(128, 64 * 36))
        self.conv = nn.Sequential(
            nn.ConvTranspose1d(64, 32, 7, padding=3), nn.ReLU(),
            nn.ConvTranspose1d(32, 16, 7, padding=3), nn.ReLU(),
            nn.ConvTranspose1d(16, 2, 7, padding=3),
        )

    def forward(self, z, y, p=None):
        h = self.fc(_gather_parts(z, y, p, self.label_embed, self.part_embed)).view(-1, 64, 36)
        return self.conv(h)

class ConvLargeKernel(AblationCVAE):
    def __init__(self, latent_dim=16, num_classes=3, embed_dim=8, condition_on_participant=False, num_participants=3, part_embed_dim=8):
        super().__init__(latent_dim, num_classes, num_participants, condition_on_participant)
        kw = dict(latent_dim=latent_dim, num_classes=num_classes, embed_dim=embed_dim, cond_part=condition_on_participant, num_participants=num_participants, part_embed_dim=part_embed_dim)
        self.encoder = _Enc_ConvLargeKernel(**kw)
        self.decoder = _Dec_ConvLargeKernel(**kw)

# conv_tiny: single Conv1d(2->8), FC 16; smallest conv variant
class _Enc_ConvTiny(nn.Module):
    def __init__(self, latent_dim, num_classes, embed_dim, cond_part, num_participants, part_embed_dim):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(2, 8, 3, padding=1), nn.ReLU())
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        self.part_embed = nn.Embedding(num_participants, part_embed_dim) if cond_part else None
        in_fc = 8 * 36 + _cond_size(embed_dim, cond_part, part_embed_dim)
        self.fc = nn.Sequential(nn.Linear(in_fc, 16), nn.ReLU())
        self.mu_head = nn.Linear(16, latent_dim)
        self.logvar_head = nn.Linear(16, latent_dim)

    def forward(self, x, y, p=None):
        h = self.conv(x).flatten(1)
        h = self.fc(_gather_parts(h, y, p, self.label_embed, self.part_embed))
        return self.mu_head(h), self.logvar_head(h)

class _Dec_ConvTiny(nn.Module):
    def __init__(self, latent_dim, num_classes, embed_dim, cond_part, num_participants, part_embed_dim):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        self.part_embed = nn.Embedding(num_participants, part_embed_dim) if cond_part else None
        in_fc = latent_dim + _cond_size(embed_dim, cond_part, part_embed_dim)
        self.fc = nn.Sequential(nn.Linear(in_fc, 16), nn.ReLU(), nn.Linear(16, 8 * 36))
        self.conv = nn.ConvTranspose1d(8, 2, 3, padding=1)

    def forward(self, z, y, p=None):
        h = self.fc(_gather_parts(z, y, p, self.label_embed, self.part_embed)).view(-1, 8, 36)
        return self.conv(h)

class ConvTiny(AblationCVAE):
    def __init__(self, latent_dim=16, num_classes=3, embed_dim=8, condition_on_participant=False, num_participants=3, part_embed_dim=8):
        super().__init__(latent_dim, num_classes, num_participants, condition_on_participant)
        kw = dict(latent_dim=latent_dim, num_classes=num_classes, embed_dim=embed_dim, cond_part=condition_on_participant, num_participants=num_participants, part_embed_dim=part_embed_dim)
        self.encoder = _Enc_ConvTiny(**kw)
        self.decoder = _Dec_ConvTiny(**kw)

# conv_slim: 2-layer Conv1d 2->8->16, FC 64
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
    def __init__(self, latent_dim=16, num_classes=3, embed_dim=8, condition_on_participant=False, num_participants=3, part_embed_dim=8):
        super().__init__(latent_dim, num_classes, num_participants, condition_on_participant)
        kw = dict(latent_dim=latent_dim, num_classes=num_classes, embed_dim=embed_dim, cond_part=condition_on_participant, num_participants=num_participants, part_embed_dim=part_embed_dim)
        self.encoder = _Enc_ConvSlim(**kw)
        self.decoder = _Dec_ConvSlim(**kw)

# mlp: no convolutions, 72->128->64->μ/σ
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
    def __init__(self, latent_dim=16, num_classes=3, embed_dim=8, condition_on_participant=False, num_participants=3, part_embed_dim=8, part_dropout=0.0):
        super().__init__(latent_dim, num_classes, num_participants, condition_on_participant, part_dropout)
        kw = dict(latent_dim=latent_dim, num_classes=num_classes, embed_dim=embed_dim, cond_part=condition_on_participant, num_participants=num_participants, part_embed_dim=part_embed_dim)
        self.encoder = _Enc_MLP(**kw)
        self.decoder = _Dec_MLP(**kw)
        # match ConvBaseline: append null-token row only when part_dropout>0 (no-op for the grid, which uses 0)
        if condition_on_participant and part_dropout > 0.0:
            rng_state = torch.get_rng_state()
            self.encoder.part_embed = _expand_embedding_with_null(self.encoder.part_embed)
            self.decoder.part_embed = _expand_embedding_with_null(self.decoder.part_embed)
            torch.set_rng_state(rng_state)

# mlp_small: 72->64->32->μ/σ
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
    def __init__(self, latent_dim=16, num_classes=3, embed_dim=8, condition_on_participant=False, num_participants=3, part_embed_dim=8):
        super().__init__(latent_dim, num_classes, num_participants, condition_on_participant)
        kw = dict(latent_dim=latent_dim, num_classes=num_classes, embed_dim=embed_dim, cond_part=condition_on_participant, num_participants=num_participants, part_embed_dim=part_embed_dim)
        self.encoder = _Enc_MLPSmall(**kw)
        self.decoder = _Dec_MLPSmall(**kw)

# mlp_tiny: single hidden layer, no nonlinearity before output projection
class _Enc_MLPTiny(nn.Module):
    def __init__(self, latent_dim, num_classes, embed_dim, cond_part, num_participants, part_embed_dim):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        self.part_embed = nn.Embedding(num_participants, part_embed_dim) if cond_part else None
        in_fc = 72 + _cond_size(embed_dim, cond_part, part_embed_dim)
        self.fc = nn.Sequential(nn.Linear(in_fc, 32), nn.ReLU())
        self.mu_head = nn.Linear(32, latent_dim)
        self.logvar_head = nn.Linear(32, latent_dim)

    def forward(self, x, y, p=None):
        h = x.flatten(1)
        h = self.fc(_gather_parts(h, y, p, self.label_embed, self.part_embed))
        return self.mu_head(h), self.logvar_head(h)

class _Dec_MLPTiny(nn.Module):
    def __init__(self, latent_dim, num_classes, embed_dim, cond_part, num_participants, part_embed_dim):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        self.part_embed = nn.Embedding(num_participants, part_embed_dim) if cond_part else None
        in_fc = latent_dim + _cond_size(embed_dim, cond_part, part_embed_dim)
        self.fc = nn.Sequential(nn.Linear(in_fc, 32), nn.Linear(32, 72))

    def forward(self, z, y, p=None):
        return self.fc(_gather_parts(z, y, p, self.label_embed, self.part_embed)).view(-1, 2, 36)

class MLPTiny(AblationCVAE):
    def __init__(self, latent_dim=16, num_classes=3, embed_dim=8, condition_on_participant=False, num_participants=3, part_embed_dim=8):
        super().__init__(latent_dim, num_classes, num_participants, condition_on_participant)
        kw = dict(latent_dim=latent_dim, num_classes=num_classes, embed_dim=embed_dim, cond_part=condition_on_participant, num_participants=num_participants, part_embed_dim=part_embed_dim)
        self.encoder = _Enc_MLPTiny(**kw)
        self.decoder = _Dec_MLPTiny(**kw)

# conv_asym: full conv encoder, single-layer decoder with dropout
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
    def __init__(self, latent_dim=16, num_classes=3, embed_dim=8, condition_on_participant=False, num_participants=3, part_embed_dim=8):
        super().__init__(latent_dim, num_classes, num_participants, condition_on_participant)
        kw = dict(latent_dim=latent_dim, num_classes=num_classes, embed_dim=embed_dim, cond_part=condition_on_participant, num_participants=num_participants, part_embed_dim=part_embed_dim)
        self.encoder = _Enc_ConvBaseline(**kw)
        self.decoder = _Dec_ConvAsym(**kw)

# conv_asym_no_dropout: weak single-layer decoder, no dropout (isolates decoder weakness)
class _Dec_ConvAsymNoDropout(nn.Module):
    def __init__(self, latent_dim, num_classes, embed_dim, cond_part, num_participants, part_embed_dim):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        self.part_embed = nn.Embedding(num_participants, part_embed_dim) if cond_part else None
        in_fc = latent_dim + _cond_size(embed_dim, cond_part, part_embed_dim)
        self.fc = nn.Sequential(nn.Linear(in_fc, 128), nn.ReLU(), nn.Linear(128, 64 * 36))
        self.conv = nn.ConvTranspose1d(64, 2, 3, padding=1)  # single deconv layer

    def forward(self, z, y, p=None):
        h = self.fc(_gather_parts(z, y, p, self.label_embed, self.part_embed)).view(-1, 64, 36)
        return self.conv(h)

class ConvAsymNoDropout(AblationCVAE):
    def __init__(self, latent_dim=16, num_classes=3, embed_dim=8, condition_on_participant=False, num_participants=3, part_embed_dim=8):
        super().__init__(latent_dim, num_classes, num_participants, condition_on_participant)
        kw = dict(latent_dim=latent_dim, num_classes=num_classes, embed_dim=embed_dim, cond_part=condition_on_participant, num_participants=num_participants, part_embed_dim=part_embed_dim)
        self.encoder = _Enc_ConvBaseline(**kw)
        self.decoder = _Dec_ConvAsymNoDropout(**kw)

# conv_baseline_dropout: full symmetric architecture, dropout in both encoder and decoder FCs
class _Enc_ConvBaselineDropout(nn.Module):
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
        self.fc = nn.Sequential(nn.Linear(in_fc, 128), nn.ReLU(), nn.Dropout(0.4))
        self.mu_head = nn.Linear(128, latent_dim)
        self.logvar_head = nn.Linear(128, latent_dim)

    def forward(self, x, y, p=None):
        h = self.conv(x).flatten(1)
        h = self.fc(_gather_parts(h, y, p, self.label_embed, self.part_embed))
        return self.mu_head(h), self.logvar_head(h)

class _Dec_ConvBaselineDropout(nn.Module):
    def __init__(self, latent_dim, num_classes, embed_dim, cond_part, num_participants, part_embed_dim):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        self.part_embed = nn.Embedding(num_participants, part_embed_dim) if cond_part else None
        in_fc = latent_dim + _cond_size(embed_dim, cond_part, part_embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(in_fc, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64 * 36),
        )
        self.conv = nn.Sequential(
            nn.ConvTranspose1d(64, 32, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(32, 16, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(16, 2, 3, padding=1),
        )

    def forward(self, z, y, p=None):
        h = self.fc(_gather_parts(z, y, p, self.label_embed, self.part_embed)).view(-1, 64, 36)
        return self.conv(h)

class ConvBaselineDropout(AblationCVAE):
    def __init__(self, latent_dim=16, num_classes=3, embed_dim=8, condition_on_participant=False, num_participants=3, part_embed_dim=8):
        super().__init__(latent_dim, num_classes, num_participants, condition_on_participant)
        kw = dict(latent_dim=latent_dim, num_classes=num_classes, embed_dim=embed_dim, cond_part=condition_on_participant, num_participants=num_participants, part_embed_dim=part_embed_dim)
        self.encoder = _Enc_ConvBaselineDropout(**kw)
        self.decoder = _Dec_ConvBaselineDropout(**kw)

# transformer: small attention-based variant; non-autoregressive seed-sequence decoder
class _Enc_Transformer(nn.Module):
    def __init__(self, latent_dim, num_classes, embed_dim, cond_part, num_participants, part_embed_dim):
        super().__init__()
        self.input_proj = nn.Linear(2, 32)
        self.pos_embed = nn.Parameter(torch.zeros(1, 36, 32))
        layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64,
                                           batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        self.part_embed = nn.Embedding(num_participants, part_embed_dim) if cond_part else None
        in_fc = 32 + _cond_size(embed_dim, cond_part, part_embed_dim)
        self.fc = nn.Sequential(nn.Linear(in_fc, 64), nn.ReLU())
        self.mu_head = nn.Linear(64, latent_dim)
        self.logvar_head = nn.Linear(64, latent_dim)

    def forward(self, x, y, p=None):
        # (B, 2, 36) -> (B, 36, 2) -> per-timestep projection to (B, 36, 32)
        h = self.input_proj(x.transpose(1, 2)) + self.pos_embed
        h = self.transformer(h).mean(dim=1)  # mean-pool over time -> (B, 32)
        h = self.fc(_gather_parts(h, y, p, self.label_embed, self.part_embed))
        return self.mu_head(h), self.logvar_head(h)

class _Dec_Transformer(nn.Module):
    def __init__(self, latent_dim, num_classes, embed_dim, cond_part, num_participants, part_embed_dim):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        self.part_embed = nn.Embedding(num_participants, part_embed_dim) if cond_part else None
        in_fc = latent_dim + _cond_size(embed_dim, cond_part, part_embed_dim)
        self.fc = nn.Linear(in_fc, 36 * 32)
        self.pos_embed = nn.Parameter(torch.zeros(1, 36, 32))
        layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.output_proj = nn.Linear(32, 2)

    def forward(self, z, y, p=None):
        h = self.fc(_gather_parts(z, y, p, self.label_embed, self.part_embed))
        h = h.view(-1, 36, 32) + self.pos_embed
        h = self.transformer(h)
        return self.output_proj(h).transpose(1, 2)  # (B, 36, 2) -> (B, 2, 36)

class Transformer(AblationCVAE):
    def __init__(self, latent_dim=16, num_classes=3, embed_dim=8, condition_on_participant=False, num_participants=3, part_embed_dim=8):
        super().__init__(latent_dim, num_classes, num_participants, condition_on_participant)
        kw = dict(latent_dim=latent_dim, num_classes=num_classes, embed_dim=embed_dim, cond_part=condition_on_participant, num_participants=num_participants, part_embed_dim=part_embed_dim)
        self.encoder = _Enc_Transformer(**kw)
        self.decoder = _Dec_Transformer(**kw)

VARIANT_MAP: dict[str, type] = {
    "conv_baseline": ConvBaseline,
    "conv_slim": ConvSlim,
    "conv_tiny": ConvTiny,
    "conv_large_kernel": ConvLargeKernel,
    "conv_asym": ConvAsym,
    "conv_asym_no_dropout": ConvAsymNoDropout,
    "conv_baseline_dropout": ConvBaselineDropout,
    "mlp": MLP,
    "mlp_small": MLPSmall,
    "mlp_tiny": MLPTiny,
    "transformer": Transformer,
}

def count_parameters(variant_name: str) -> int:
    model = VARIANT_MAP[variant_name]()
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
