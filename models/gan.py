import torch
import torch.nn as nn


class Generator(nn.Module):
    """Conditional 1D conv generator: (z, class[, participant]) -> (B, 2, 36) in z-scored space.
    Participant embedding has a +1 null row for LOSO generation of an unseen subject."""
    def __init__(self, z_dim=16, num_classes=3, num_participants=5, embed_dim=8, part_embed_dim=8,
                 condition_on_participant=True, part_dropout=0.0, hidden=64):
        super().__init__()
        self._cond_part = condition_on_participant
        self.z_dim = z_dim
        self.hidden = hidden
        self.num_participants = num_participants
        self.null_part_idx = num_participants
        self.part_dropout = part_dropout
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        self.part_embed = nn.Embedding(num_participants + 1, part_embed_dim) if condition_on_participant else None
        in_dim = z_dim + embed_dim + (part_embed_dim if condition_on_participant else 0)
        h2, h4 = hidden // 2, hidden // 4
        self.fc = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, hidden * 36), nn.ReLU())
        self.conv = nn.Sequential(nn.ConvTranspose1d(hidden, h2, 3, padding=1), nn.ReLU(),
                                  nn.ConvTranspose1d(h2, h4, 3, padding=1), nn.ReLU(),
                                  nn.ConvTranspose1d(h4, 2, 3, padding=1))

    def forward(self, z, y, p=None):
        parts = [z, self.label_embed(y)]
        if self._cond_part and p is not None:
            parts.append(self.part_embed(p))
        h = self.fc(torch.cat(parts, dim=1)).view(-1, self.hidden, 36)
        return self.conv(h)

    @torch.no_grad()
    def sample(self, n, y, device, participant: int | None = None):
        if y.dim() == 0:
            y = y.expand(n)
        z = torch.randn(n, self.z_dim, device=device)
        if not self._cond_part:
            p = None
        elif participant is not None:
            p = torch.full((n,), participant, dtype=torch.long, device=device)
        else:
            p = torch.randint(0, self.num_participants, (n,), device=device)
        self.eval()
        return self.forward(z, y.to(device), p)


class Discriminator(nn.Module):
    """Class-conditional 1D conv discriminator: (x, class) -> real/fake logit. (Conditions on class only;
    participant conditioning lives in the generator so LOSO null-token generation stays clean.)"""
    def __init__(self, num_classes=3, embed_dim=8, hidden=64):
        super().__init__()
        h2, h4 = hidden // 2, hidden // 4
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        self.conv = nn.Sequential(nn.Conv1d(2, h4, 3, padding=1), nn.LeakyReLU(0.2),
                                  nn.Conv1d(h4, h2, 3, padding=1), nn.LeakyReLU(0.2))
        self.fc = nn.Sequential(nn.Linear(h2 * 36 + embed_dim, 128), nn.LeakyReLU(0.2), nn.Dropout(0.3),
                                nn.Linear(128, 1))

    def forward(self, x, y):
        h = self.conv(x).flatten(1)
        return self.fc(torch.cat([h, self.label_embed(y)], dim=1))
