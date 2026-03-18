import torch
from tqdm import tqdm
import numpy as np
import os, sys, copy
import time
from torch.autograd import grad
from pytorch_msssim import ssim

sys.path.append(os.path.abspath('../modules'))
import utility as ut
import vae_loss as vl
import vae_train as vt
import vae_ortho as vo
import vae_viz as viz


@torch.no_grad()
def collect_latents(net, dataloader, device=None, use_mu=True, max_batches=None):
    """
    Collect latent vectors from a dataloader using the VAE encoder.

    Parameters
    ----------
    net : nn.Module
        VAE model with encoder returning (mu, logvar) or equivalent through net(x).
    dataloader : torch.utils.data.DataLoader
        Dataloader yielding (images, labels) or compatible tuples.
    device : torch.device or None
        Device to run on. If None, uses net.device if available.
    use_mu : bool
        If True, use encoder mean mu as latent representation.
        If False, sample z = mu + std * eps.
    max_batches : int or None
        If not None, stop after this many batches.

    Returns
    -------
    latents : torch.Tensor, shape [N, latent_dim]
        Collected latent codes.
    """
    if device is None:
        device = getattr(net, "device", next(net.parameters()).device)

    net.eval()
    all_latents = []

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Collecting latents", leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break

        # support (img, label) or other variants
        imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
        imgs = imgs.view(imgs.shape[0], -1).to(device)

        _, mu, logvar = net(imgs)

        if use_mu:
            z = mu
        else:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std

        all_latents.append(z.detach().cpu())

    if not all_latents:
        raise ValueError("No latents were collected. Check dataloader/max_batches.")

    return torch.cat(all_latents, dim=0)


def compute_feature_direction_and_threshold(z_retain, z_forget, eps=1e-12):
    """
    Compute feature direction z_e and threshold delta.

    Parameters
    ----------
    z_forget : torch.Tensor, shape [Nf, d]
        Latents for the forget / positive set.
    z_retain : torch.Tensor, shape [Nr, d]
        Latents for the retain / negative set.
    eps : float
        Small number for numerical stability.

    Returns
    -------
    result : dict
        {
            "mu_forget": ...,
            "mu_retain": ...,
            "feature_direction": z_e,
            "feature_direction_unit": v_unit,
            "forget_projections": proj_forget,
            "retain_projections": proj_retain,
            "forget_projection_mean": a_forget,
            "retain_projection_mean": a_retain,
            "delta": delta,
        }
    """
    if z_forget.ndim != 2 or z_retain.ndim != 2:
        raise ValueError("z_forget and z_retain must both be 2D tensors of shape [N, latent_dim].")

    if z_forget.shape[1] != z_retain.shape[1]:
        raise ValueError("Latent dimensions must match.")

    mu_forget = z_forget.mean(dim=0)
    mu_retain = z_retain.mean(dim=0)

    # feature direction
    z_e = mu_forget - mu_retain

    # normalize for projection computations
    v_norm = torch.norm(z_e) + eps
    v_unit = z_e / v_norm

    # scalar projections onto the feature direction
    proj_forget = z_forget @ v_unit
    proj_retain = z_retain @ v_unit

    a_forget = proj_forget.mean()
    a_retain = proj_retain.mean()

    # midpoint threshold
    delta = 0.5 * (a_forget + a_retain)

    return {
        "mu_forget": mu_forget,
        "mu_retain": mu_retain,
        "feature_direction": z_e,
        "feature_direction_unit": v_unit,
        "forget_projections": proj_forget,
        "retain_projections": proj_retain,
        "forget_projection_mean": a_forget,
        "retain_projection_mean": a_retain,
        "delta": delta,
    }


@torch.no_grad()
def compute_direction_and_threshold_from_dataloaders(
    net,
    retain_loader,
    forget_loader,
    device=None,
    use_mu=True,
    max_batches=None,
):
    """
    Convenience wrapper:
    - collect forget latents
    - collect retain latents
    - compute feature direction and delta

    Parameters
    ----------
    net : nn.Module
        Trained VAE.
    forget_loader : DataLoader
        Loader for forget / positive data.
    retain_loader : DataLoader
        Loader for retain / negative data.
    device : torch.device or None
        Device to use.
    use_mu : bool
        Whether to use encoder mean mu as latent.
    max_batches : int or None
        Optional cap on batches per loader.

    Returns
    -------
    result : dict
        Output of compute_feature_direction_and_threshold plus latent tensors.
    """
    z_forget = collect_latents(
        net=net,
        dataloader=forget_loader,
        device=device,
        use_mu=use_mu,
        max_batches=max_batches,
    )

    z_retain = collect_latents(
        net=net,
        dataloader=retain_loader,
        device=device,
        use_mu=use_mu,
        max_batches=max_batches,
    )

    result = compute_feature_direction_and_threshold(z_retain, z_forget)
    # result["z_forget"] = z_forget
    # result["z_retain"] = z_retain
    return result


def project_onto_direction(z, v_unit):
    """
    Scalar projection of latent(s) z onto unit direction v_unit.

    Parameters
    ----------
    z : torch.Tensor, shape [..., d]
    v_unit : torch.Tensor, shape [d]

    Returns
    -------
    proj : torch.Tensor, shape [...]
    """
    return z @ v_unit


def sim(z, v_unit, delta):
    """
    Classifier c(z) = 1{proj(z) > delta}

    Parameters
    ----------
    z : torch.Tensor, shape [N, d] or [d]
    v_unit : torch.Tensor, shape [d]
    delta : float or scalar tensor

    Returns
    -------
    mask : torch.FloatTensor
    """
    return (project_onto_direction(z, v_unit) > delta).float()




@torch.no_grad()
def compute_z_hat(z, z_e, v_unit, delta):
    """
    Compute edited latent:
        z_hat = z - max(proj_vt(z) - delta, 0) * z_e

    Assumes:
        z     : [B, latent_dim]
        z_e   : [latent_dim]
        v_unit : [latent_dim], unit vector along z_e
        delta : scalar
    """
    # z_e = z_e.to(z.device)
    # delta = torch.as_tensor(delta, device=z.device, dtype=z.dtype)

    # # normalize direction for stable projections
    # v_unit = z_e / (torch.norm(z_e) + eps)

    # scalar projections: [B]
    proj = project_onto_direction(z, v_unit)

    # only remove excess over delta
    excess = torch.clamp(proj - delta, min=0.0)

    # subtract along feature direction
    z_hat = z - excess.unsqueeze(1) * z_e.unsqueeze(0)
    return z_hat

def L_recon(g, f, z, v_unit, delta):
    """
    Paper-style L_recon for a VAE.

    This function applies the piecewise rule batchwise.

    Parameters
    ----------
    g : nn.Module
        Trainable VAE. Assumed to return reconstructed, mu, logvar on net(x).
    f : nn.Module
        Frozen reference VAE with decoder used as D0.
    v_unit : torch.Tensor
        Feature direction normalized, shape [latent_dim].
    delta : float or scalar tensor
        Threshold.


    Returns
    -------
    loss : torch.Tensor

    """
    gz = g.decoder(z)
    fz = f.decoder(z)
    s = sim(z, v_unit, delta)
    return ((1 - s) * torch.abs(gz - fz).sum(dim=1)).mean()


def L_full(g, f, z, z_e, v_unit, delta, alpha=3.):
    z_hat = compute_z_hat(z, z_e, v_unit, delta)
    gz = g.decoder(z)
    fz = f.decoder(z)
    fzhat = f.decoder(z_hat)
    s = sim(z, v_unit, delta)
    l_recon = ((1 - s) * torch.abs(gz - fz).sum(dim=1)).mean()
    l_unlearn = (s * torch.abs(gz - fzhat).sum(dim=1)).mean()
    l_percep = (s * (1. - ssim(gz.view(-1, 1, 28, 28), fzhat.view(-1, 1, 28, 28), data_range=1.0))).mean()

    return l_recon + alpha * (l_unlearn + l_percep)




def get_processor(net, net0, identifier, z_random, z_e, v_unit, delta, weights, optim, all_digits, forget_digit):
    """
    Returns a function that processes a batch of images through a VAE network and computes the necessary gradients.

    This function performs a forward and backward pass on a batch of images, calculating the reconstruction loss, 
    KL divergence, and a uniformity loss. It then adjusts the gradients to ensure orthogonality, performs an 
    optimization step, and returns the relevant metrics.

    Parameters:
    net (nn.Module): The VAE model.
    net0 (nn.Module): The frozen reference VAE model.
    identifier (nn.Module): The model used for logits computation.
    z_random (torch.tensor): Random latent codes for the decoder.
    z_e (torch.tensor): Feature direction.
    v_unit (torch.tensor): Unit vector along the feature direction.
    delta (float): Threshold.
    weights (tuple): Contains weights for KL divergence and uniformity loss.
    optim (torch.optim.Optimizer): Optimizer for the VAE.
    all_digits (list): List of all class labels.
    forget_digit (int): The class label to forget.

    Returns:
    function: A function that takes a batch of images to retain and forget, and returns the reconstruction loss, 
              KL divergence, uniformity loss, orthogonality measure, generated image, logits, and elapsed time.
    """
    def process_batch(real_img_retain, real_img_forget):
        kl_weight, uniformity_weight = weights
        z = torch.randn(2*real_img_retain.shape[0], net.latent_dim).to(net.device)
        time_0 = time.time()
        optim.zero_grad()

        l_full = L_full(net, net0, z, z_e, v_unit, delta, alpha=3.)
        l_full.backward()

        optim.step()
        time_final = time.time() 

        elapsed_time = time_final - time_0

        with torch.no_grad():
            generated_img = net.decoder(z_random)
            logits = identifier(generated_img)
            uniformity = vl.uniformity_loss_surgery(logits, all_digits=all_digits, forget_digit=forget_digit)

        return None, None, uniformity.item(), None, generated_img, logits, elapsed_time

    return process_batch






def train(model, folder, num_steps, batch_size, latent_dim=2, save_steps=None, collect_interval='epoch', log_interval=10,\
          kl_weight=1., uniformity_weight=1e4, all_digits=list(range(10)), forget_digit=1,\
          img_ext='jpg', classifier_path="../data/MNIST/classifiers/MNISTClassifier.pth", data_path='../../data/MNIST', **viz_kwargs):
    """
    Train the VAE on MNIST digits, with a custom loop to alternate between ascent and descent steps, using the "surgery" method to orthogonalize the gradients.

    Parameters
    ----------
    model : str or torch.nn.Module
        Path to a saved model or a model itself.
    folder : str
        Folder to store results.
    num_steps : int
        Number of training steps.
    batch_size : int
        Batch size for training.
    latent_dim : int, optional
        Dimensionality of the latent space. Defaults to 2.
    save_steps : int or None, optional
        Interval at which to save the model. Defaults to None, which means to never save.
    collect_interval : str, optional
        Interval at which to collect samples. Must be 'epoch', 'step', or None. Defaults to 'epoch'.
    log_interval : int, optional
        Interval at which to log results. Defaults to 10.
    kl_weight : float, optional
        Weight for the KL loss. Defaults to 1.
    uniformity_weight : float, optional
        Weight for the uniformity loss. Defaults to 1e4.
    all_digits : list, optional
        List of all digits to use. Defaults to list(range(10)).
    forget_digit : int, optional

        Digit to forget. Defaults to 1.
    img_ext : str, optional
        Extension to use for saved images. Defaults to 'jpg'.
    classifier_path : str, optional
        Path to a saved classifier. Defaults to "../data/MNIST/classifiers/MNISTClassifier.pth".
    **viz_kwargs : dict, optional
        Additional keyword arguments to pass to `viz.summarize_training`.

    Returns
    -------
    None
    """
    # ---------------------------------------------------
    # Setup
    # ---------------------------------------------------
    net, dataloader, optim, z_random, identifier, sample_dir, checkpoint_dir, epoch_length, epochs,\
    num_steps, save_steps, collect_interval, log_interval, csv_file, device, grid_size \
    = vt.init(model, folder, num_steps, batch_size, latent_dim=latent_dim, save_steps=save_steps, collect_interval=collect_interval,\
           log_interval=log_interval, kl_weight=kl_weight, uniformity_weight=uniformity_weight, orthogonality_weight=0.,\
           all_digits=all_digits, forget_digit=forget_digit, img_ext=img_ext, classifier_path=classifier_path, train_mode='orthogonal', data_path=data_path)
    
    log_results = vo.get_logger(identifier, csv_file, log_interval)
    save = vt.get_saver(net, save_steps, checkpoint_dir, epoch_length)
    collect_samples = vt.get_collector(sample_dir, collect_interval, grid_size, img_ext) 
    net0 = ut.freeze_all(copy.deepcopy(net).eval()).to(device)
    res = compute_direction_and_threshold_from_dataloaders(net0, dataloader['retain'], dataloader['forget'], device, use_mu=True)
    z_e = res['feature_direction'].to(device)
    v_unit = res['feature_direction_unit'].to(device)
    delta = res['delta']

    process_batch = get_processor(net, net0, identifier, z_random, z_e, v_unit, delta, (kl_weight, uniformity_weight), optim, all_digits, forget_digit)    
    # ---------------------------------------------------
    # Main training loop
    # ---------------------------------------------------
    global_step = 0
    for _ in tqdm(range(1, epochs + 1), desc="Epochs"):
        for (img_retain, _), (img_forget, _) in zip(dataloader['retain'], dataloader['forget']):
            global_step += 1
            # -- Process a single batch
            rec_loss, kl_loss, unif_loss, orth_loss, generated_img, logits, elapsed_time = process_batch(img_retain, img_forget)
            loss = None#rec_loss + kl_weight * kl_loss + uniformity_weight * unif_loss 
            real_img, _ = next(iter(dataloader['original']))
            real_img = real_img.view(real_img.shape[0], -1).to(device)
            log_results(step=global_step, losses=[rec_loss, kl_loss, unif_loss, orth_loss, loss], elapsed_time=elapsed_time, real_img=real_img, generated_img=generated_img, logits=logits)
            save(step=global_step)
            collect_samples(generated_img, step=global_step)
    viz_kwargs.update({"folder": folder})
    viz.summarize_training(**viz_kwargs) 




