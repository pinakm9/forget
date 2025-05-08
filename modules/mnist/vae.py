import torch 
import torch.nn as nn
import os
import sys


# Add the 'modules' directory to the Python search path
sys.path.append(os.path.abspath('../modules'))
import utility as ut
import lora


# Define VAE Model
class VAE(nn.Module):
    def __init__(self, latent_dim = 2, device="cuda"):
        """
        Initialize the VAE model.

        Parameters
        ----------
        latent_dim : int, optional
            The dimensionality of the latent space. Default is 2.
        device : str, optional
            The device to run the model on. Default is 'cuda'.
        """
        super(VAE, self).__init__()
        self.device = device

        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(nn.Linear(784, 400),
                                     nn.BatchNorm1d(400),
                                     nn.ReLU()
                                     )
                                     
                                     
        self.enc_log_sigma = nn.Linear(400, self.latent_dim)
        self.enc_mu = nn.Linear(400, self.latent_dim)
        
        self.decoder = nn.Sequential(nn.Linear(self.latent_dim, 400),
                                     nn.BatchNorm1d(400),
                                     nn.ReLU(),
                                     nn.Linear(400, 784),
                                     nn.Sigmoid()
                                     )

    def encode(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            The input tensor to be encoded

        Returns
        -------
        mu : torch.Tensor
            The mean of the latent variables
        logvar : torch.Tensor
            The log variance of the latent variables
        """
        h1 = self.encoder(x)
        return self.enc_mu(h1), self.enc_log_sigma(h1)
        
    def sample_latent(self, mu, log_sigma):
        """
        Parameters
        ----------
        mu : torch.Tensor
            The mean of the latent variables
        logvar : torch.Tensor
            The log variance of the latent variables

        Returns
        -------
        latent_sample : torch.Tensor
            A sample from the latent variables
        """
        sigma = torch.exp(0.5*log_sigma).to(self.device)
        
        eps = torch.Tensor(sigma.shape).to(self.device).normal_()
        return eps.mul(sigma).add_(mu)
        
    def forward(self, input):
        """
        Parameters
        ----------
        input : torch.Tensor
            The input tensor to be transformed

        Returns
        -------
        output : torch.Tensor
            The output tensor
        mu : torch.Tensor
            The mean of the latent variables
        logvar : torch.Tensor
            The log variance of the latent variables
        """
        mu, logvar = self.encode(input)
        
        z = self.sample_latent(mu, logvar)
        return self.decoder(z), mu, logvar
    
    def freeze_encoder(self):
        ut.freeze_all(self.encoder)
    



class LoRA_VAE(nn.Module):
    def __init__(self, frozen_vae: VAE, lora_r=4, lora_alpha=1.0):
        """
        Initializes a LoRA-adapted VAE model using an already frozen VAE.
        
        Parameters
        ----------
        frozen_vae : VAE
            A pretrained and frozen VAE model.
        lora_r : int, optional
            Rank of the low-rank update in LoRA layers. Default is 4.
        lora_alpha : float, optional
            Scaling factor for the LoRA updates. Default is 1.0.
        """
        super(LoRA_VAE, self).__init__()
        self.latent_dim = frozen_vae.latent_dim
        self.device = frozen_vae.device

        # --- Encoder ---
        # Convert the encoder's linear layer to LoRA version.
        encoder_linear = frozen_vae.encoder[0]  # Assuming the first module is nn.Linear.
        self.encoder = nn.Sequential(
            lora.convert_linear_to_lora(encoder_linear, lora_r, lora_alpha),
            frozen_vae.encoder[1],  # BatchNorm remains the same.
            frozen_vae.encoder[2]   # ReLU remains the same.
        )

        # Convert the mu and log_sigma layers.
        self.enc_mu = lora.convert_linear_to_lora(frozen_vae.enc_mu, lora_r, lora_alpha)
        self.enc_log_sigma = lora.convert_linear_to_lora(frozen_vae.enc_log_sigma, lora_r, lora_alpha)

        # --- Decoder ---
        # The decoder is assumed to be a Sequential of five modules:
        # [Linear, BatchNorm, ReLU, Linear, Sigmoid]
        decoder_linear1 = frozen_vae.decoder[0]
        decoder_linear2 = frozen_vae.decoder[3]
        self.decoder = nn.Sequential(
            lora.convert_linear_to_lora(decoder_linear1, lora_r, lora_alpha),
            frozen_vae.decoder[1],  # BatchNorm
            frozen_vae.decoder[2],  # ReLU
            lora.convert_linear_to_lora(decoder_linear2, lora_r, lora_alpha),
            frozen_vae.decoder[4]   # Sigmoid
        )

    def reparameterize(self, mu, log_sigma):
        std = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = self.encoder(x)
        mu = self.enc_mu(h)
        log_sigma = self.enc_log_sigma(h)
        z = self.reparameterize(mu, log_sigma)
        x_recon = self.decoder(z)
        return x_recon, mu, log_sigma
    



class LoRA_VAE_Decoder(nn.Module):
    def __init__(self, frozen_vae: VAE, lora_r=4, lora_alpha=1.0):
        """
        Initializes a LoRA-adapted VAE model using an already frozen VAE.
        LoRA is applied only on the decoder; the encoder remains as in the frozen model.
        
        Parameters
        ----------
        frozen_vae : VAE
            A pretrained and frozen VAE model.
        lora_r : int, optional
            Rank of the low-rank update in LoRA layers. Default is 4.
        lora_alpha : float, optional
            Scaling factor for the LoRA updates. Default is 1.0.
        """
        super(LoRA_VAE_Decoder, self).__init__()
        self.latent_dim = frozen_vae.latent_dim
        self.device = frozen_vae.device

        # --- Encoder ---
        # Keep the encoder intact (without LoRA modifications).
        self.encoder = frozen_vae.encoder
        self.enc_mu = frozen_vae.enc_mu 
        self.enc_log_sigma = frozen_vae.enc_log_sigma

        # --- Decoder ---
        # The decoder is assumed to be a Sequential of five modules:
        # [Linear, BatchNorm, ReLU, Linear, Sigmoid]
        # Apply LoRA only on the linear layers of the decoder.
        decoder_linear1 = frozen_vae.decoder[0]
        decoder_linear2 = frozen_vae.decoder[3]
        self.decoder = nn.Sequential(
            lora.convert_linear_to_lora(decoder_linear1, lora_r, lora_alpha),
            frozen_vae.decoder[1],  # BatchNorm remains the same.
            frozen_vae.decoder[2],  # ReLU remains the same.
            lora.convert_linear_to_lora(decoder_linear2, lora_r, lora_alpha),
            frozen_vae.decoder[4]   # Sigmoid remains the same.
        )

    def reparameterize(self, mu, log_sigma):
        std = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = self.encoder(x)
        mu = self.enc_mu(h)
        log_sigma = self.enc_log_sigma(h)
        z = self.reparameterize(mu, log_sigma)
        x_recon = self.decoder(z)
        return x_recon, mu, log_sigma





