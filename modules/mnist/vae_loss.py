import torch

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
    return torch.nn.functional.binary_cross_entropy(input, output, reduction='sum') 


def loss(input, output, mu, logvar, beta=1.):
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
    return reconstruction_loss(input, output) + beta*kl_div(mu, logvar)



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
    return torch.nn.functional.binary_cross_entropy(input, output, reduction='sum') / input.shape[0]


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