import torch
from torchvision.utils import save_image
from tqdm import tqdm


# Rejection sampling function
def sample_glasses_latents(G, C, latent_dim=512, batch_size=64, num_samples=100, threshold=0.5, device='cpu'):
    """
    Perform rejection sampling to generate latent vectors that produce images classified as having glasses.

    Parameters
    ----------
    G : torch.nn.Module
        The generator model used to decode latent vectors into images.
    C : torch.nn.Module
        The classifier model used to determine if the generated images have glasses.
    latent_dim : int, optional
        The dimensionality of the latent space. Default is 100.
    batch_size : int, optional
        The number of samples to generate per batch. Default is 64.
    num_samples : int, optional
        The total number of latent vectors to sample. Default is 100.
    threshold : float, optional
        The threshold for classifying images as having glasses. Default is 0.5.
    device : str, optional
        The device on which to perform computation ('cpu' or 'cuda'). Default is 'cpu'.

    Returns
    -------
    torch.Tensor
        A tensor of shape (num_samples, latent_dim) containing the accepted latent vectors.
    """

    G.eval()
    C.eval()
    
    accepted_z = []
    with torch.no_grad():
        pbar = tqdm(total=num_samples)
        while len(accepted_z) < num_samples:
            z = torch.randn(batch_size, latent_dim, device=device)
            generated = G.decode(z)  # Assume output is in [0, 1] or normalized to match C input
            
            logits = C(generated)  # Output shape: (batch_size,)
            preds = torch.sigmoid(logits)

            for i in range(batch_size):
                if preds[i].item() > threshold:
                    accepted_z.append(z[i].unsqueeze(0))
                    pbar.update(1)
                    if len(accepted_z) >= num_samples:
                        break
        pbar.close()
    
    return torch.cat(accepted_z, dim=0)


