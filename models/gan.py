import torch
import torch.nn as nn

class Generator(nn.Module):
    """
    Conditional generator: maps noise z and class label y to a synthetic signal.
    Mirrors the ConditionalDecoder from the CVAE for architectural consistency.
    """
    def __init__(self, latent_dim: int = 16, num_classes: int = 3, embed_dim: int = 8) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        self.fc = nn.Sequential(nn.Linear(latent_dim + embed_dim, 128),
                                nn.ReLU(),
                                nn.Linear(128, 64 * 36),
                                nn.ReLU())
        self.conv = nn.Sequential(nn.ConvTranspose1d(64, 32, kernel_size=3, padding=1),
                                  nn.ReLU(),
                                  nn.ConvTranspose1d(32, 16, kernel_size=3, padding=1),
                                  nn.ReLU(),
                                  nn.ConvTranspose1d(16, 2, kernel_size=3, padding=1))

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        :param z: (B, latent_dim) noise vectors
        :param y: (B,) integer class labels
        :return: (B, 2, 36) generated signals
        """
        e = self.label_embed(y) # (B, embed_dim)
        h = self.fc(torch.cat([z, e], dim=1)) # (B, 64*36)
        h = h.view(h.size(0), 64, 36) # (B, 64, 36)
        return self.conv(h) # (B, 2, 36)

class Discriminator(nn.Module):
    """
    Conditional discriminator: classifies signals as real or fake, conditioned on class label.
    The class label is embedded and projected to signal length 36, then concatenated as an extra channel.
    Uses LeakyReLU (standard for discriminators) and outputs a raw logit (no sigmoid).
    """
    def __init__(self, num_classes: int = 3, embed_dim: int = 8) -> None:
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        self.label_proj  = nn.Linear(embed_dim, 36)
        # 3 input channels: 2 signal channels + 1 projected label channel
        self.conv = nn.Sequential(nn.Conv1d(3, 16, kernel_size=3, padding=1),
                                  nn.LeakyReLU(0.2),
                                  nn.Conv1d(16, 32, kernel_size=3, padding=1),
                                  nn.LeakyReLU(0.2),
                                  nn.Conv1d(32, 64, kernel_size=3, padding=1),
                                  nn.LeakyReLU(0.2))
        self.fc = nn.Sequential(nn.Linear(64 * 36, 128),
                                nn.LeakyReLU(0.2),
                                nn.Linear(128, 1))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B, 2, 36) signals (real or fake)
        :param y: (B,) integer class labels
        :return: (B, 1) raw logits (positive = real)
        """
        e         = self.label_embed(y) # (B, embed_dim)
        label_map = self.label_proj(e).unsqueeze(1) # (B, 1, 36)
        xc        = torch.cat([x, label_map], dim=1)# (B, 3, 36)
        h         = self.conv(xc).flatten(1) # (B, 64*36)
        return self.fc(h) # (B, 1)

class CGAN(nn.Module):
    """Wraps Generator and Discriminator into a Conditional GAN."""
    def __init__(self, latent_dim: int = 16, num_classes: int = 3, embed_dim: int = 8) -> None:
        super().__init__()
        self.latent_dim  = latent_dim
        self.num_classes = num_classes
        self.generator     = Generator(latent_dim, num_classes, embed_dim)
        self.discriminator = Discriminator(num_classes, embed_dim)

    def sample(self, n: int, y: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Sample n signals conditioned on class labels y.

        :param n: number of samples to generate
        :param y: (n,) integer class labels (or scalar broadcast to all n)
        :param device: device to perform sampling on
        :return: (n, 2, 36) generated signals
        """
        z = torch.randn(n, self.latent_dim, device=device)
        if y.dim() == 0:
            y = y.expand(n)
        self.generator.eval()
        with torch.no_grad():
            return self.generator(z, y.to(device))

def gradient_penalty(discriminator: nn.Module, real: torch.Tensor, fake: torch.Tensor,
                      y: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    WGAN-GP gradient penalty: enforces 1-Lipschitz constraint on the critic by penalising
    gradients that deviate from norm 1 on interpolated samples.
    """
    B = real.size(0)
    alpha = torch.rand(B, 1, 1, device=device) # (B, 1, 1) broadcast over (B, 2, 36)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp = discriminator(interp, y)
    grads = torch.autograd.grad(outputs=d_interp, inputs=interp,
                                grad_outputs=torch.ones_like(d_interp),
                                create_graph=True, retain_graph=True)[0]
    grads = grads.reshape(B, -1)
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()

def discriminator_loss(real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
    """Wasserstein critic loss: maximise E[real] - E[fake]."""
    return fake_scores.mean() - real_scores.mean()

def generator_loss(fake_scores: torch.Tensor) -> torch.Tensor:
    """Wasserstein generator loss: maximise E[fake]."""
    return -fake_scores.mean()
