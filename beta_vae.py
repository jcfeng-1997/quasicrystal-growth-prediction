import torch
from torch import nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, latent_dim, input_channels=1, input_shape=(256, 256)):
        super().__init__()

        self.latent_dim = latent_dim
        self.input_channels = input_channels
        self.input_shape = input_shape

        self.encoder_conv = self.buildEncoderConv()
        self.flatten_dim = self._get_flatten_dim()

        self.encoder_base = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, 256),
            nn.ELU(),
        )

        self.fc_mu_logvar = nn.Linear(256, latent_dim * 2)
        nn.init.normal_(self.fc_mu_logvar.weight, mean=0.0, std=0.01) 
        nn.init.constant_(self.fc_mu_logvar.bias, 0)

        self.decoder = self.buildDecoder(latent_dim)

    def buildEncoderConv(self):
        return nn.Sequential(
            nn.Conv2d(self.input_channels, 8, 3, 2, 1), nn.ELU(),
            nn.Conv2d(8, 16, 3, 2, 1), nn.ELU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ELU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ELU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.ELU(),
        )

    def _get_flatten_dim(self):
        with torch.no_grad():
            dummy = torch.zeros(1, self.input_channels, *self.input_shape)
            out = self.encoder_conv(dummy)
            return out.view(1, -1).shape[1]

    def buildDecoder(self, latent_dim):
        return nn.Sequential(
            nn.Linear(latent_dim, 512), nn.ELU(),                     
            nn.Linear(512, 256 * 4 * 4), nn.ELU(),
            nn.Unflatten(dim=1, unflattened_size=(256, 4, 4)),

            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1), nn.ELU(),                               

            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1), nn.ELU(),                              

            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1), nn.ELU(),
            nn.ConvTranspose2d(32, 16, 3, 2, 1, 1), nn.ELU(),
            nn.ConvTranspose2d(16, 8, 3, 2, 1, 1), nn.ELU(),
            nn.ConvTranspose2d(8, 1, 3, 2, 1, 1), 
            nn.Sigmoid() 
        )

    def sample(self, mean, logvariance):
        std = torch.exp(0.5 * logvariance)  
        epsilon = torch.randn_like(std)
        return mean + epsilon * std

    def forward(self, data):
        x = self.encoder_conv(data)
        x = self.encoder_base(x)
        mean_logvar = self.fc_mu_logvar(x)
        mean, logvariance = torch.chunk(mean_logvar, 2, dim=1)
        z = self.sample(mean, logvariance)
        reconstruction = self.decoder(z)
        return reconstruction, mean, logvariance

