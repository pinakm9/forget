import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import tqdm
import sys, os
import torch.nn.functional as F
from scipy.linalg import sqrtm
import numpy as np


# Add the 'modules' directory to the Python search path
sys.path.append(os.path.abspath('../modules'))
import utility as ut



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x)
    


def count_male(images, model="../../data/CelebA/cnn/cnn_10.pth", device="cuda"):
    """
    Count the number of images that are classified as male using a pre-trained CNN model.

    Parameters:
        images (torch.Tensor): A tensor containing images with shape (N, 3, 64, 64).
        model (str or nn.Module): The path to a pre-trained CNN model or the model instance itself.
        device (str): The device on which to run the model ('cuda' or 'cpu').

    Returns:
        int: The number of images that are classified as male.
    """
    if isinstance(model, str):
        model = torch.load(model)
    model.to(device)
    # Set the model to evaluation mode.
    model.eval()
    # Move the images to the specified device.
    images = images.to(device)

    # Disable gradient calculation for inference.
    with torch.no_grad():
        logits = model(images)
        # Get the raw logits from the model.
        # Determine predicted classes by taking the argmax over the logits.
        predictions = (torch.sigmoid(logits) > 0.5).float()
    
    # Count the number of predictions that equal the specified digit.
    count = (predictions == 1).sum().item()
    return count


def get_classifier(path="../../data/CelebA/cnn/cnn_10.pth", device="cuda"):
    """
    Load a pre-trained CNN classifier from a checkpoint file and return it as an instance of nn.Module.
    
    Parameters:
        path (str): The path to the checkpoint file (default: '../../data/CelebA/cnn/cnn_10.pth').
        device (str): The device on which to run the model ('cuda' or 'cpu').
        
    Returns:
        nn.Module: An instance of nn.Module with the pre-trained weights.
    """
    model = CNN()
    model.load_state_dict(torch.load(path, weights_only=True, map_location=device))
    model.to(device)
    model.eval()
    return model

def count_from_logits(logits):
    """
    Count the number of logits classified as positive (greater than 0.5 after applying sigmoid).

    Parameters:
        logits (torch.Tensor): The input logits.

    Returns:
        int: The number of logits classified as positive.
    """

    return (torch.sigmoid(logits) > 0.5).float().sum().item()

def entropy(probs):
    """Computes entropy-based uncertainty from probabilities."""
    return -(probs * torch.log(probs + 1e-9) + (1 - probs) * torch.log(1 - probs + 1e-9))  # Avoid log(0)

def margin_confidence(probs):
    """Computes margin-based uncertainty (inverted: lower margin means more ambiguous) from probabilities."""
    margin = torch.abs(2. * probs - 1.)
    return 1 - margin  # Invert: low margin = high ambiguity

def ambiguity(logits, weight_entropy=0.5, weight_margin=0.5):
    """Combines entropy and margin-based ambiguity detection."""
    probs = torch.sigmoid(logits) 
    entropy_scores = entropy(probs)
    margin_scores = margin_confidence(probs)
    
    # Normalize scores to [0, 1] (optional for better scaling)
    entropy_scores = (entropy_scores - entropy_scores.min()) / (entropy_scores.max() - entropy_scores.min() + 1e-9)
    margin_scores = (margin_scores - margin_scores.min()) / (margin_scores.max() - margin_scores.min() + 1e-9)
    
    # Weighted combination
    # combined_score = weight_entropy * entropy_scores + weight_margin * margin_scores
    return entropy_scores, margin_scores#, combined_score


def compute_features(images, identifier, layer=None):
    """
    Extract features from a single image or batch using a forward hook.

    Args:
        identifier (nn.Module): identifier instance (e.g., CNN)
        images (Tensor): input image tensor, shape [C, H, W] or [B, C, H, W]
        layer (nn.Module, optional): target layer to hook. Defaults to identifier.classifier[1].

    Returns:
        np.ndarray: feature vector(s), shape [1, D] or [B, D]
    """
    features = []

    if layer is None:
        layer = identifier.classifier[1]  # default: dense layer before output

    def hook_fn(module, input, output):
        features.append(output.detach().cpu())

    hook = layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = identifier(images)

    hook.remove()
    return features[0].numpy()


# @ut.timer
def frechet_inception_distance(real_images, gen_images, identifier):
    """
    Computes the Frechet Inception Distance (FID) between two sets of images using NumPy.
    
    FID is computed as the squared difference between the mean feature vectors of the two sets,
    plus the trace of the sum of the covariance matrices minus twice the matrix square root 
    of their product.
    
    Parameters
    ----------
    real_images : array-like
        Real images.
    gen_images : array-like
        gen images.
    identifier : callable
        A function or network that computes features from images. It should accept an image (or batch)
        and return a NumPy array of features.
        
    Returns
    -------
    fid : float
        The FID between the real and gen images.
    """
    # # Extract features from real and gen images
    # real_features = compute_features(real_images, identifier)  # shape: (N, D)
    # gen_features = compute_features(gen_images, identifier) # shape: (M, D)
    
    

    # mu_real, sigma_real = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    # mu_gen, sigma_gen = gen_features.mean(axis=0), np.cov(gen_features, rowvar=False)


    # cov_sqrt = sqrtm(sigma_real @ sigma_gen)

    # if np.iscomplexobj(cov_sqrt):
    #     cov_sqrt = cov_sqrt.real

    # # Compute FID score
    # fid = np.sum((mu_real - mu_gen) ** 2) + np.trace(sigma_real + sigma_gen - 2 * cov_sqrt)

    # if fid < 0:
    #     sigma_real += np.eye(sigma_real.shape[0]) * 1e-8
    #     sigma_gen += np.eye(sigma_gen.shape[0]) * 1e-8
    #     if np.iscomplexobj(cov_sqrt):
    #         cov_sqrt = cov_sqrt.real
    #     # Compute FID score
    #     fid = np.sum((mu_real - mu_gen) ** 2) + np.trace(sigma_real + sigma_gen - 2 * cov_sqrt)

    # return fid
     # 1) Extract features
    real_feats = compute_features(real_images, identifier)  # (N, D)
    gen_feats  = compute_features(gen_images,  identifier)  # (M, D)

    # 2) Compute statistics
    mu_r = real_feats.mean(axis=0)
    mu_g = gen_feats.mean(axis=0)
    cov_r = np.cov(real_feats, rowvar=False)
    cov_g = np.cov(gen_feats,  rowvar=False)

    # 3) Symmetrize covariances
    cov_r = (cov_r + cov_r.T) / 2
    cov_g = (cov_g + cov_g.T) / 2

    # 4) Compute sqrt of product, with retries
    covmean = None
    for i in range(2):
        try:
            covmean, _ = sqrtm(cov_r @ cov_g, disp=False)
            if not np.isfinite(covmean).all():
                raise ValueError("Non-finite result")
            break
        except Exception:
            # on failure add jitter and retry
            jitter = np.eye(cov_r.shape[0]) * (1e-8 * (10**i))
            cov_r_j = cov_r + jitter
            cov_g_j = cov_g + jitter
            covmean, _ = sqrtm(cov_r_j @ cov_g_j, disp=False)

    # 5) Clean up sqrtm output
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    covmean = (covmean + covmean.T) / 2  # re-symmetrize

    # 6) Final FID formula
    diff = mu_r - mu_g
    fid = diff @ diff + np.trace(cov_r + cov_g - 2 * covmean)
    return float(np.real(fid))

def inception_score(logits):
    probs = torch.sigmoid(logits)
    eps = 1e-9
    # Convert probs to [p0, p1] per sample
    probs = torch.stack([1 - probs, probs], dim=1)  # shape: [batch_size, 2]

    # Marginal probability p(y)
    py = torch.mean(probs, dim=0)  # shape: [2]

    # KL divergence per sample: sum_i p_i * (log p_i - log py_i)
    kl_divs = torch.sum(probs * (torch.log(probs + eps) - torch.log(py + eps)), dim=1)

    # Inception Score = exp(mean(KL))
    return torch.exp(kl_divs.mean())



