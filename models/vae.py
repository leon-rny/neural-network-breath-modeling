import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    Compress a raw signal (B, 2, 36) into a compact description of its distribution in latent space.
    Convolutional layers: first layer extracts simple features, later layers combine those into more complex patterns.
    Fully connected bottleneck: maps the extracted features into a latent distribution (mu, logvar).
    """
    def __init__(self, latent_dim: int = 16) -> None:
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
        """
        Defines forward pass through encoder

        :param x: (B, 2, 36) input signals
        :return: (B, latent_dim) mean and log-variance of latent distribution
        """
        h = self.conv(x) # (B, 64, 36)
        h = h.flatten(1) # (B, 64*36)
        h = self.fc(h) # (B, 128)
        return self.mu_head(h), self.logvar_head(h)

class Decoder(nn.Module):
    """
    Mirror of the encoder that maps a latent vector back to the original signal space.
    """
    def __init__(self, latent_dim: int = 16) -> None:
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
        """
        Defines forward pass through decoder

        :param z: (B, latent_dim) latent vectors
        :return: (B, 2, 36) reconstructed signals
        """
        h = self.fc(z) # (B, 64*36)
        h = h.view(h.size(0), 64, 36) # (B, 64, 36)
        return self.conv(h) # (B, 2, 36)

class VAE(nn.Module):
    """Combines the encoder and decoder, implements the reparameterization trick and defines a sampling method."""
    def __init__(self, latent_dim: int = 16) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: during training, sample from the latent distribution using mu and logvar.
        
        :param mu: (B, latent_dim) mean of the latent distribution
        :param logvar: (B, latent_dim) log-variance of the latent distribution
        :return: (B, latent_dim) sampled latent vector
        """
        if self.training:
            std = (0.5 * logvar).exp()
            return mu + std * torch.randn_like(std)
        return mu

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VAE: encode input, sample from latent distribution, decode to reconstruct.

        :param x: (B, 2, 36) input signals
        :return: (B, 2, 36) reconstructed signals, (B, latent_dim) mean, (B, latent_dim) log-variance
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        """
        Sample n signals from the prior N(0, I).
        
        :param n: number of samples to generate
        :param device: device to perform sampling on
        :return: (n, 2, 36) generated signals
        """
        z = torch.randn(n, self.latent_dim, device=device)
        self.eval()
        with torch.no_grad():
            return self.decoder(z)

class ConditionalEncoder(nn.Module):
    """
    Conditional variant of Encoder: conditions the latent distribution on a class label.
    The class label is embedded and concatenated to the flattened conv features before the FC bottleneck.
    """
    def __init__(self, latent_dim: int = 16, num_classes: int = 3, embed_dim: int = 8) -> None:
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(2, 16, kernel_size=3, padding=1),
                                  nn.ReLU(),
                                  nn.Conv1d(16, 32, kernel_size=3, padding=1),
                                  nn.ReLU(),
                                  nn.Conv1d(32, 64, kernel_size=3, padding=1),
                                  nn.ReLU())
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        self.fc = nn.Sequential(nn.Linear(64 * 36 + embed_dim, 128),
                                nn.ReLU())
        self.mu_head = nn.Linear(128, latent_dim)
        self.logvar_head = nn.Linear(128, latent_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: (B, 2, 36) input signals
        :param y: (B,) integer class labels
        :return: (B, latent_dim) mean and log-variance
        """
        h = self.conv(x).flatten(1) # (B, 64*36)
        e = self.label_embed(y) # (B, embed_dim)
        h = self.fc(torch.cat([h, e], dim=1)) # (B, 128)
        return self.mu_head(h), self.logvar_head(h)

class ConditionalDecoder(nn.Module):
    """Conditional variant of Decoder: the latent vector is concatenated with a class embedding before reconstruction."""
    def __init__(self, latent_dim: int = 16, num_classes: int = 3, embed_dim: int = 8) -> None:
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        self.fc = nn.Sequential(nn.Linear(latent_dim + embed_dim, 128),
                                nn.ReLU(),
                                nn.Linear(128, 64 * 36))
        self.conv = nn.Sequential(nn.ConvTranspose1d(64, 32, kernel_size=3, padding=1),
                                  nn.ReLU(),
                                  nn.ConvTranspose1d(32, 16, kernel_size=3, padding=1),
                                  nn.ReLU(),
                                  nn.ConvTranspose1d(16, 2, kernel_size=3, padding=1))

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        :param z: (B, latent_dim) latent vectors
        :param y: (B,) integer class labels
        :return: (B, 2, 36) reconstructed signals
        """
        e = self.label_embed(y)                    # (B, embed_dim)
        h = self.fc(torch.cat([z, e], dim=1))      # (B, 64*36)
        h = h.view(h.size(0), 64, 36)              # (B, 64, 36)
        return self.conv(h)                        # (B, 2, 36)

class CVAE(nn.Module):
    """Conditional VAE: wraps ConditionalEncoder and ConditionalDecoder, conditioning on class label."""
    def __init__(self, latent_dim: int = 16, num_classes: int = 3, embed_dim: int = 8) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.encoder = ConditionalEncoder(latent_dim, num_classes, embed_dim)
        self.decoder = ConditionalDecoder(latent_dim, num_classes, embed_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick (same as VAE)."""
        if self.training:
            std = (0.5 * logvar).exp()
            return mu + std * torch.randn_like(std)
        return mu

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param x: (B, 2, 36) input signals
        :param y: (B,) integer class labels
        :return: (B, 2, 36) reconstructed signals, (B, latent_dim) mu, (B, latent_dim) logvar
        """
        mu, logvar = self.encoder(x, y)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z, y)
        return x_hat, mu, logvar

    def sample(self, n: int, y: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Sample n signals conditioned on class labels y.

        :param n: number of samples to generate
        :param y: (n,) integer class labels (or scalar broadcast to all n samples)
        :param device: device to perform sampling on
        :return: (n, 2, 36) generated signals
        """
        z = torch.randn(n, self.latent_dim, device=device)
        if y.dim() == 0:
            y = y.expand(n)
        self.eval()
        with torch.no_grad():
            return self.decoder(z, y.to(device))

def elbo_loss(x: torch.Tensor, x_hat: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ELBO loss = MSE reconstruction + beta * KL divergence.

    :param x: (B, 2, 36) original signals
    :param x_hat: (B, 2, 36) reconstructed signals
    :param mu: (B, latent_dim) mean of latent distribution
    :param logvar: (B, latent_dim) log-variance of latent distribution
    :param beta: weight for KL divergence (for annealing)
    :return: total loss, reconstruction loss, KL divergence
    """
    recon = nn.functional.mse_loss(x_hat, x, reduction='mean')
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
    return recon + beta * kl, recon, kl
