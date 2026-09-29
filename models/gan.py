import torch
from torch import nn


class Generator(nn.Module):
    """Conditional 1D conv generator: (z, class[, participant]) -> (B, 2, 36) in z-scored space.
    Participant embedding has a +1 null row for LOSO generation of an unseen subject."""
    def __init__(self, z_dim=16, num_classes=3, num_participants=5, embed_dim=8, part_embed_dim=8, condition_on_participant=True, part_dropout=0.0, hidden=64):
        """
        :param z_dim: Dimension of the input noise vector (default 16).
        :param num_classes: Number of breath classes to condition on (default 3).
        :param num_participants: Number of participants (default 5); a +1 null row is added for LOSO generation.
        :param embed_dim: Dimension of the class embedding (default 8).
        :param part_embed_dim: Dimension of the participant embedding (default 8).
        :param condition_on_participant: Whether to condition on participant id (default True).
        :param part_dropout: Probability of dropping participant conditioning to the null token during training (default 0.0).
        :param hidden: Hidden channel width of the conv stack (default 64).
        """
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
        """
        :param z: Tensor of shape (B, z_dim) containing the input noise vectors.
        :param y: Tensor of shape (B,) containing the class labels to condition on.
        :param p: Optional tensor of shape (B,) containing the participant labels to condition on. If None, no participant conditioning will be applied.
        :return: Tensor of shape (B, 2, 36) containing the generated signals in z-scored space.
        """
        parts = [z, self.label_embed(y)]
        if self._cond_part and p is not None:
            parts.append(self.part_embed(p))
        h = self.fc(torch.cat(parts, dim=1)).view(-1, self.hidden, 36)
        return self.conv(h)

    @torch.no_grad()
    def sample(self, n, y, device, participant: int | None = None):
        """
        :param n: Number of samples to generate.
        :param y: Tensor of shape (n,) or scalar containing the class labels to condition on.
        :param device: The device to generate the samples on.
        :param participant: Optional int specifying the participant to condition on. If None, a random participant will be sampled for each generated sample. Use null_part_idx for LOSO unseen-subject generation.
        :return: Tensor of shape (n, 2, 36) containing the generated samples in z-scored space.
        """
        if y.dim() == 0:
            y = y.expand(n)
        z = torch.randn(n, self.z_dim, device=device)
        if not self._cond_part:
            p = None
        elif participant is not None:
            p = torch.full((n,), participant, dtype=torch.long, device=device)
        else:
            pool = torch.tensor(getattr(self, 'trained_participants', list(range(self.num_participants))), device=device)
            p = pool[torch.randint(len(pool), (n,), device=device)]
        self.eval()
        return self.forward(z, y.to(device), p)

class Discriminator(nn.Module):
    """Class-conditional 1D conv discriminator: (x, class) -> real/fake logit. (Conditions on class only;
    participant conditioning lives in the generator so LOSO null-token generation stays clean.)"""
    def __init__(self, num_classes=3, embed_dim=8, hidden=64):
        """
        :param num_classes: Number of breath classes to condition on (default 3).
        :param embed_dim: Dimension of the class embedding (default 8).
        :param hidden: Hidden channel width of the conv stack (default 64).
        """
        super().__init__()
        h2, h4 = hidden // 2, hidden // 4
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        self.conv = nn.Sequential(nn.Conv1d(2, h4, 3, padding=1), nn.LeakyReLU(0.2),
                                  nn.Conv1d(h4, h2, 3, padding=1), nn.LeakyReLU(0.2))
        self.fc = nn.Sequential(nn.Linear(h2 * 36 + embed_dim, 128), nn.LeakyReLU(0.2), nn.Dropout(0.3),
                                nn.Linear(128, 1))

    def forward(self, x, y):
        """
        :param x: Tensor of shape (B, 2, 36) containing the input signals in z-scored space.
        :param y: Tensor of shape (B,) containing the class labels to condition on.
        :return: Tensor of shape (B, 1) containing the real/fake logits
        """
        h = self.conv(x).flatten(1)
        return self.fc(torch.cat([h, self.label_embed(y)], dim=1))
