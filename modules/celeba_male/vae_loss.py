import torch
import os, sys
from torch.autograd import grad

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

