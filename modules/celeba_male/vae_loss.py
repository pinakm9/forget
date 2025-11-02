import torch
import torch.nn as nn
from pytorch_msssim import ms_ssim, MS_SSIM
import os, sys
from torch.autograd import grad
import torch.nn.functional as F

sys.path.append(os.path.abspath('../modules'))
import utility as ut

def kl_div(mu, logvar):
    """
    Compute the Kullback-Leibler divergence between the Gaussian distribution
    specified by mu and logvar, and a standard normal distribution.

    Parameters
    ----------
    mu : torch.Tensor
        The mean of the latent variables.
    logvar : torch.Tensor
        The log variance of the latent variables.

    Returns
    -------
    torch.Tensor
        The KL divergence loss.
    """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def reconstruction_loss(input, output):
    """
    Compute the reconstruction loss between the input and output tensors.

    Parameters
    ----------
    input : torch.Tensor
        The input tensor.
    output : torch.Tensor
        The output tensor.

    Returns
    -------
    torch.Tensor
        The reconstruction loss.
    """
    return torch.nn.functional.mse_loss(input, output, reduction='sum') 

def loss(input, output, mu, logvar, kl_weight=1.):
    return reconstruction_loss(input, output) + kl_weight*kl_div(mu, logvar)

def mean_kl_div(mu, logvar):
    """
    Compute the Kullback-Leibler divergence loss averaged over the batch.
    
    The KL divergence is computed for each sample by summing over the latent dimensions,
    and then the average over all samples is returned.
    
    Parameters
    ----------
    mu : torch.Tensor
        The mean of the latent variables (shape: [batch_size, latent_dim]).
    logvar : torch.Tensor
        The log variance of the latent variables (shape: [batch_size, latent_dim]).
    
    Returns
    -------
    torch.Tensor
        The KL divergence loss averaged over the batch.
    """
    # Sum over latent dimensions for each sample
    # Average over the batch
    return torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


def mean_reconstruction_loss(input, output):
    """
    Compute the reconstruction loss between the input and output tensors.

    Parameters
    ----------
    input : torch.Tensor
        The input tensor.
    output : torch.Tensor
        The output tensor.

    Returns
    -------
    torch.Tensor
        The reconstruction loss.
    """
    return torch.nn.functional.mse_loss(input, output, reduction='sum') / input.shape[0]

def mean_loss(input, output, mu, logvar, beta=1.):
    """
    Compute the total loss of the VAE, which is the sum of the reconstruction loss and the KL divergence loss.

    Parameters
    ----------
    input : torch.Tensor
        The input tensor.
    output : torch.Tensor
        The output tensor.
    mu : torch.Tensor
        The mean of the latent variables (shape: [batch_size, latent_dim]).
    logvar : torch.Tensor
        The log variance of the latent variables (shape: [batch_size, latent_dim]).
    beta : float, optional
        The weight of the KL divergence term in the loss. Default is 1.

    Returns
    -------
    torch.Tensor
        The total loss of the VAE.
    """
    return mean_reconstruction_loss(input, output) + beta*mean_kl_div(mu, logvar)


def uniformity_loss(logits, classes=[0, 1]):    
    """
    Compute the uniformity loss over the given logits.

    Parameters
    ----------
    logits : torch.Tensor
        The input logits (shape: [batch_size]).
    classes : list of int
        The list of classes to consider.

    Returns
    -------
    float
        The uniformity loss.
    """
    probs = torch.sigmoid(logits).mean(dim=0)
    # Convert probs to [p0, p1] per sample
    probs = torch.stack([1 - probs, probs], dim=0)  # shape: [2]
    mask = torch.zeros_like(probs)
    mask[classes] = 1.
    uniform_target = mask/len(classes)
    return torch.sum(probs * torch.log((probs + 1e-8) / (uniform_target + 1e-8))) 


def uniformity_loss_surgery(logits, all_classes=[0, 1], forget_class=1):    
    """
    Compute the uniformity loss over the given logits, but forget forget_class.

    Parameters
    ----------
    logits : torch.Tensor
        The input logits (shape: [batch_size]).
    all_classes : list of int, optional
        The list of all classes to consider. Default is [0, 1].
    forget_class : int, optional
        The class to forget (i.e., not to include in the uniformity loss). Default is 1.

    Returns
    -------
    float
        The uniformity loss.
    """
    probs = torch.sigmoid(logits).mean(dim=0)
    # Convert probs to [p0, p1] per sample
    probs = torch.stack([1 - probs, probs], dim=0)  # shape: [2]
    mask = torch.zeros_like(probs)
    mask[all_classes] = 1.
    mask[forget_class] = 0
    uniform_target = mask/(len(all_classes) - 1)
    return torch.sum(probs * torch.log((probs + 1e-8) / (uniform_target + 1e-8)))

# @ut.timer
def orthogonality_loss(model, identifier, retain_sample, forget_sample, kl_weight=1., uniformity_weight=0., classes=[0, 1], latent_dim=512):
    """
    Computes the orthogonality loss between the gradients of the loss with respect to the forget and retain samples.

    Parameters
    ----------
    model : VAE
        The VAE model.
    identifier : Identifier
        The identifier network.
    retain_sample : torch.Tensor
        The retain sample.
    forget_sample : torch.Tensor
        The forget sample.
    kl_weight : float
        The weight of the KL divergence term in the loss.
    uniformity_weight : float
        The weight of the uniformity loss term.
    classes : list of int
        The classes to consider in the uniformity loss.

    Returns
    -------
    float
        The orthogonality loss.
    """
    
    trainable_params = ut.get_trainable_params(model)

    reconstructed_forget, mu_forget, logvar_forget = model(forget_sample)
    reconstructed_retain, mu_retain, logvar_retain = model(retain_sample)

    rec_forget = reconstruction_loss(reconstructed_forget, forget_sample)
    rec_retain = reconstruction_loss(reconstructed_retain, retain_sample)
    kl_forget = kl_div(mu_forget, logvar_forget)
    kl_retain = kl_div(mu_retain, logvar_retain)

    generated_img = model.decode(torch.randn(retain_sample.shape[0], latent_dim).to(next(model.parameters()).device))
    logits = identifier(generated_img)
    uniformity = uniformity_loss(logits, classes)

    loss_forget = rec_forget + kl_weight * kl_forget + uniformity_weight * uniformity
    loss_retain = rec_retain + kl_weight * kl_retain + uniformity_weight * uniformity

    gf = torch.cat([x.view(-1) for x in grad(outputs=loss_forget, inputs=trainable_params, retain_graph=True)])
    gr = torch.cat([x.view(-1) for x in grad(outputs=loss_retain, inputs=trainable_params)])

    return (torch.sum(gf * gr)**2 / (torch.sum(gf * gf) * torch.sum(gr * gr)))



class MSSSIMLoss(nn.Module):
    def __init__(self, data_range=1.0, size_average=True, channels=3, levels=3):
        super().__init__()
        self.ms_ssim = MS_SSIM(
            data_range=data_range,
            size_average=size_average,
            channel=channels,
            weights=None,     # default weights
            levels=levels     # number of downsampling levels
        )

    def forward(self, x, y):
        """
        Computes MS-SSIM loss for inputs in [0, 1], shape [B, C, H, W]
        """
        return 1 - self.ms_ssim(x, y)
    

def proj(z: torch.Tensor, z_e: torch.Tensor) -> torch.Tensor:
    """
    Project each z in batch onto direction z_e.

    Parameters:
        z (torch.Tensor): Latent codes, shape [B, D]
        z_e (torch.Tensor): Feature direction, shape [D]

    Returns:
        torch.Tensor: Projection scalars, shape [B]
    """
    return torch.matmul(z, z_e) / torch.dot(z_e, z_e)


def hat(z: torch.Tensor, z_e: torch.Tensor, t: float = 0.0) -> torch.Tensor:
    """
    Computes z_hat = z - (proj(z, z_e) - t) * z_e for a batch of z.

    Parameters:
        z (torch.Tensor): Latents, shape [B, D]
        z_e (torch.Tensor): Feature direction, shape [D]
        t (float): Scalar threshold (default 0.0)

    Returns:
        torch.Tensor: Transformed latents, shape [B, D]
    """
    # shape [B]
    projection = proj(z, z_e)  # scalar projection per z_i

    # shape [B, 1] for broadcasting
    scale = (projection - t).unsqueeze(1)  # shape [B, 1]

    # shape [B, D] - broadcasting z_e over batch
    return z - scale * z_e



def sim(z, z_e, t=0):
    return (proj(z, z_e) > t).float()



def ssim(img1, img2, C1=0.01**2, C2=0.03**2, window_size=11):
    """
    Compute per-sample SSIM for a batch of images.
    Inputs:
        img1, img2: [B, C, H, W] in [0, 1]
    Returns:
        Tensor of shape [B] — SSIM score for each sample
    """
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size // 2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size // 2)

    sigma1_sq = F.avg_pool2d(img1 ** 2, window_size, stride=1, padding=window_size // 2) - mu1 ** 2
    sigma2_sq = F.avg_pool2d(img2 ** 2, window_size, stride=1, padding=window_size // 2) - mu2 ** 2
    sigma12   = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size // 2) - mu1 * mu2

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean(dim=[1, 2, 3])  # per-sample SSIM




def per_sample_ms_ssim_64(img1, img2, levels=3, weights=None):
    """
    Custom MS-SSIM for 64x64 images — returns per-sample scores.
    
    Inputs:
        img1, img2: [B, C, H, W] in [0, 1]
        levels: number of downscaling steps (max 3 for 64x64)
        weights: list of floats (optional); should sum to 1

    Output:
        Tensor of shape [B] — MS-SSIM score for each sample
    """
    if weights is None:
        weights = [0.3, 0.3, 0.4]
    assert len(weights) == levels
    assert levels <= 3, "Only up to 3 levels supported for 64x64 images"

    B = img1.shape[0]
    msssim = torch.ones(B, device=img1.device)

    for i in range(levels):
        ssim_i = ssim(img1, img2)  # [B]
        msssim *= ssim_i ** weights[i]

        # Downsample for next scale
        if i < levels - 1:
            img1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
            img2 = F.avg_pool2d(img2, kernel_size=2, stride=2)

    return msssim  # shape [B]


class PerSampleMSSSIMLoss(nn.Module):
    def __init__(self, levels=3, weights=None):
        super().__init__()
        self.levels = levels
        self.weights = weights or [0.3, 0.3, 0.4]

    def forward(self, x, y):
        """
        Inputs:
            x, y: [B, C, H, W], in [0, 1]
        Returns:
            [B] loss values = 1 - MS-SSIM per sample
        """
        return 1.0 - per_sample_ms_ssim_64(x, y, self.levels, self.weights)
    


def L_percep(x, y, s):
    return s * (1.0 - per_sample_ms_ssim_64(x, y, 3, [0.3, 0.3, 0.4])) 


def L_unlearn(x, y, s):
    return s * torch.norm(x - y, p=1, dim=[1, 2, 3])

def L_recon(x, y, s):
    return (1 - s) * torch.norm(x - y, p=1, dim=[1, 2, 3])