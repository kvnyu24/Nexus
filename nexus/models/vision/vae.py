import torch
import torch.nn as nn
import torch.nn.functional as F
from ...core.base import NexusModule

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        hidden = F.relu(self.linear(x))
        mu = self.mu(hidden)
        log_var = self.log_var(hidden)
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(latent_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        hidden = F.relu(self.linear(z))
        reconstructed = torch.sigmoid(self.out(hidden))
        return reconstructed

class VAE(NexusModule):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = Encoder(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            latent_dim=config['latent_dim']
        )
        self.decoder = Decoder(
            latent_dim=config['latent_dim'],
            hidden_dim=config['hidden_dim'],
            output_dim=config['input_dim']
        )
        self.beta = config.get('beta', 1.0)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decoder(z)
        return {
            'reconstruction': reconstruction,
            'mu': mu,
            'log_var': log_var,
            'z': z
        }

    def decode(self, z):
        return self.decoder(z) 