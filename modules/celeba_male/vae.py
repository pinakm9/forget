import torch 
import torch.nn as nn
import os
import sys


# Add the 'modules' directory to the Python search path
sys.path.append(os.path.abspath('../modules'))
import utility as ut

IMAGE_SIZE = 64


class VAE(nn.Module):
    def __init__(self, latent_dim=512, in_channels=3):
        """
        Initialize the VAE model.

        Parameters
        ----------
        latent_dim : int, optional
            The dimensionality of the latent space. Default is 512.
        in_channels : int, optional
            The number of input channels. Default is 3.
        """
        super().__init__()
        # self.latent_dim = latent_dim
        # Encoder hidden dims
        encoder_hidden_dims = [32, 64, 128, 256, 512]
        modules = []
        for h_dim in encoder_hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)

        # Compute flattened features size
        num_downsamples = len(encoder_hidden_dims)
        feature_map_size = IMAGE_SIZE // (2 ** num_downsamples)
        flattened_size = encoder_hidden_dims[-1] * feature_map_size * feature_map_size

        # Latent vectors mu and logvar
        self.fc_mu  = nn.Linear(flattened_size, latent_dim)
        self.fc_var = nn.Linear(flattened_size, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, flattened_size)
        decoder_hidden_dims = encoder_hidden_dims[::-1]
        modules = []
        for i in range(len(decoder_hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        decoder_hidden_dims[i],
                        decoder_hidden_dims[i+1],
                        kernel_size=4, stride=2, padding=1
                    ),
                    nn.BatchNorm2d(decoder_hidden_dims[i+1]),
                    nn.LeakyReLU()
                )
            )
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                decoder_hidden_dims[-1],
                decoder_hidden_dims[-1],
                kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(decoder_hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(decoder_hidden_dims[-1], 3, kernel_size=3, padding=1),
            nn.Tanh()  # outputs in [-1,1] to match normalized inputs
        )

    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu     = self.fc_mu(x)
        logvar = self.fc_var(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, self.encoder[-1][0].out_channels,
                   IMAGE_SIZE // (2 ** len(self.encoder)),
                   IMAGE_SIZE // (2 ** len(self.encoder)))
        x = self.decoder(x)
        return self.final_layer(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar