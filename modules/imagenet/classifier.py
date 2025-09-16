from transformers import AutoImageProcessor, AutoModelForImageClassification
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from scipy.linalg import sqrtm
import torch, torch.nn.functional as F
from contextlib import nullcontext
import gc

def get_classifier(device):
    """
    Return a pre-trained SwinV2 image classification model.

    Parameters:
        device (str): The device on which to run the model ('cuda' or 'cpu').

    Returns:
        torch.nn.Module: A pre-trained SwinV2 image classification model.
    """
    model = AutoModelForImageClassification.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256").to(device)
    model.eval()
    return model 


def count_from_logits(logits, class_id):
    """
    Count the number of logits classified as a specific class.

    Parameters:
        logits (torch.Tensor): The input logits.
        class_id (int): The class to count (e.g., 1 for counting ones).

    Returns:
        int: The number of logits classified as the specified class.
    """
    return (logits.argmax(dim=-1) == class_id).sum().item()  


def count_class(images, device, class_id):
    """
    Count the number of images classified as a specific class.

    Parameters:
        images (torch.Tensor): A tensor containing digit images with shape (N, 1, 28, 28).
        device (str): The device on which to run the model ('cuda' or 'cpu').
        class_id (int): The class to count (e.g., 1 for counting ones).

    Returns:
        int: The number of images that are classified as the specified class.
    """
    model = get_classifier(device)
    images = images.to(device)
    logits = model(images).logits
    return count_from_logits(logits, class_id)




def _best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

@torch.no_grad()
def compute_features(images, batch_size=64, device=None):
    device = _best_device() if device is None else torch.device(device)

    # lazy-init per device
    ex = getattr(compute_features, "_extractor", None)
    ex_dev = getattr(compute_features, "_dev", None)
    if ex is None or ex_dev != str(device):
        w = models.Inception_V3_Weights.IMAGENET1K_V1
        m = models.inception_v3(weights=w).eval().to(device)  # uses transform_input=True internally
        ex = create_feature_extractor(m, return_nodes={"avgpool": "pool"})
        ex.requires_grad_(False)
        compute_features._extractor, compute_features._dev = ex, str(device)

    # ---- standardize to NCHW float in [0,1], but DO NOT mean/std-normalize
    x = images
    if x.dim() != 4:
        raise ValueError(f"Expected 4D, got {x.shape}")
    x = x.float()
    if x.max() > 1.5:                    # [0,255] -> [0,1]
        x = x / 255.0
    if x.min() < 0:                      # [-1,1]   -> [0,1]
        x = (x + 1.0) / 2.0

    # resize to 299x299; let the model's internal transform_input handle channel scaling
    x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)

    feats = []
    nb = device.type == "cuda"
    for i in range(0, x.size(0), batch_size):
        xb = x[i:i+batch_size].to(device, non_blocking=nb)
        out = ex(xb)["pool"].squeeze(-1).squeeze(-1)  # (B, 2048)
        feats.append(out.cpu())
    return torch.cat(feats, 0)







def FID(real_imgs, gen_imgs,
        compute_features=compute_features,
        eps: float = 1e-6, batch_size: int = 64, device='cuda') -> float:
    """
    Computes the Frechet Inception Distance (FID) between two sets of images.

    Parameters
    ----------
    real_imgs : torch.Tensor
        The real images to use for FID computation.
    gen_imgs : torch.Tensor
        The generated images to use for FID computation.
    compute_features : callable, optional
        A function that computes features from images. Defaults to `compute_features`.
    eps : float, optional
        A small value to add to the covariance matrices for numerical stability. Defaults to 1e-6.
    batch_size : int, optional
        The batch size to use for feature extraction. Defaults to 64.
    device : str, optional
        The device to use for feature extraction. Defaults to 'cuda'.

    Returns
    -------
    float
        The FID score between the real and generated images.
    """
     # 1) Extract features (-> numpy)
    f_r = compute_features(real_imgs, batch_size, device).detach().cpu().numpy()
    f_g = compute_features(gen_imgs, batch_size, device).detach().cpu().numpy()

    # 2) Means & covariances
    mu_r = np.mean(f_r, axis=0)
    mu_g = np.mean(f_g, axis=0)
    sigma_r = np.cov(f_r, rowvar=False)
    sigma_g = np.cov(f_g, rowvar=False)

    # 3) sqrtm(sigma_r * sigma_g)
    I = np.eye(sigma_r.shape[0], dtype=np.float64)
    A = (sigma_r + eps * I) @ (sigma_g + eps * I)
    try:
        covmean = sqrtm(A)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
    except Exception:
        # Eigen fallback if SciPy isn't available
        A = 0.5 * (A + A.T)
        vals, vecs = np.linalg.eigh(A)
        vals = np.clip(vals, 0.0, None)
        covmean = (vecs * np.sqrt(vals)) @ vecs.T

    # 4) FID formula
    diff = mu_r - mu_g
    fid = diff @ diff + np.trace(sigma_r + sigma_g - 2.0 * covmean)
    return float(fid)









def identify(identifier, gen_imgs, forget_class, device):
    """
    FP16 inference on CUDA/MPS, FP32 elsewhere. Safe on Apple M-series (MPS).
    Returns: class_count in [0,1].
    """

    # Pick an autocast context per device
    if device.type == "cuda":
        amp_ctx = torch.autocast("cuda", dtype=torch.float16)
    elif device.type == "mps":
        # MPS autocast supports float16; bf16 is not generally available on MPS
        amp_ctx = torch.autocast("mps", dtype=torch.float16)
    else:
        amp_ctx = nullcontext()   # CPU: no autocast

    with torch.no_grad(), amp_ctx:
        count = (identifier(gen_imgs).logits.argmax(dim=-1) == forget_class).sum().item() 
    # Do downstream ops in float32 on CPU for numerical safety
    return count
