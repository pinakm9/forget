import os
import sys
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm


sys.path.append(os.path.abspath('../modules'))
import vae_loss as vl
import vae_ortho as vo
import vae_train as vt
import vae_viz as viz


def _freeze_batchnorm_stats(model):
    """Keep BatchNorm affine parameters trainable without updating its buffers."""
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def _classifier_logits(identifier, images):
    """Run the classifier for evaluation only, never for the training loss."""
    if images.ndim == 2 and images.shape[1] == 28 * 28:
        images = images.reshape(images.shape[0], 1, 28, 28)
    return identifier(images)


def L_unlearn(
    model,
    x_forget,
    x_retain,
    kl_weight=1.0,
    retain_weight=1.0,
):
    """Paper-inspired generative SALUN objective for an unconditional VAE.

    Conditional SALUN minimizes the MSE between the model prediction under the
    forgotten concept ``c`` and a randomly selected, misaligned concept ``c'``.
    An unconditional VAE has no concept input, so a randomly paired retain
    example supplies the closest available surrogate for ``c'``. Both examples
    use the same latent-noise realization before decoding, mirroring the shared
    noisy input used by the conditional diffusion objective.

    The second term is the ordinary retained-data VAE objective, corresponding
    to the retained diffusion-training loss in the paper. No classifier or
    frozen teacher is part of this loss.
    """
    mu_forget, logvar_forget = model.encode(x_forget)
    mu_retain, logvar_retain = model.encode(x_retain)

    # Randomly pair each forget posterior with a retain posterior. A shared
    # epsilon makes their difference represent the concept/data pairing rather
    # than two unrelated reparameterization-noise draws.
    pair_count = min(mu_forget.shape[0], mu_retain.shape[0])
    if pair_count == 0:
        raise ValueError("Forget and retain batches must both be non-empty")
    retain_indices = torch.randperm(
        mu_retain.shape[0], device=mu_retain.device
    )[:pair_count]
    shared_epsilon = torch.randn_like(mu_forget[:pair_count])
    z_forget = (
        mu_forget[:pair_count]
        + shared_epsilon * torch.exp(0.5 * logvar_forget[:pair_count])
    )
    z_misaligned = (
        mu_retain[retain_indices]
        + shared_epsilon
        * torch.exp(0.5 * logvar_retain[retain_indices])
    )
    prediction_forget = model.decoder(z_forget)
    prediction_misaligned = model.decoder(z_misaligned)
    forget_loss = F.mse_loss(prediction_forget, prediction_misaligned)

    # Normal VAE training objective on D_r. Dividing the complete per-example
    # ELBO by the number of pixels preserves its optimum and its reconstruction
    # versus KL balance while making it comparable to pixel-mean MSE above.
    retain_epsilon = torch.randn_like(mu_retain)
    z_retain = (
        mu_retain + retain_epsilon * torch.exp(0.5 * logvar_retain)
    )
    reconstructed_retain = model.decoder(z_retain)
    pixels_per_image = x_retain[0].numel()
    retain_reconstruction = (
        vl.mean_reconstruction_loss(reconstructed_retain, x_retain)
        / pixels_per_image
    )
    retain_kl = vl.mean_kl_div(mu_retain, logvar_retain) / pixels_per_image
    retain_loss = retain_reconstruction + kl_weight * retain_kl

    # Paper form: L_misaligned + beta * L_retain.
    total_loss = forget_loss + retain_weight * retain_loss
    components = {
        "forget": forget_loss,
        "retain_reconstruction": retain_reconstruction,
        "retain_kl": retain_kl,
        "retain": retain_loss,
    }
    return total_loss, components


def get_processor(
    net,
    identifier,
    z_random,
    weights,
    optim,
    mask,
    max_grad_norm=5.0,
):
    """Create one masked, numerically guarded unlearning step."""
    kl_weight, retain_weight = weights

    theta_0 = {name: parameter.detach().clone()
               for name, parameter in net.named_parameters()}
    buffers_0 = {name: buffer.detach().clone()
                 for name, buffer in net.named_buffers()}

    # The classifier remains available for the existing generation metrics, but
    # it is not called until after the optimization step under no_grad.
    identifier.eval()
    identifier.requires_grad_(False)

    net.train()
    _freeze_batchnorm_stats(net)

    def process_batch(real_img_retain, real_img_forget):
        time_0 = time.time()
        optim.zero_grad(set_to_none=True)

        real_img_forget = real_img_forget.reshape(
            real_img_forget.shape[0], -1
        ).to(net.device)
        real_img_retain = real_img_retain.reshape(
            real_img_retain.shape[0], -1
        ).to(net.device)

        loss, components = L_unlearn(
            net,
            real_img_forget,
            real_img_retain,
            kl_weight=kl_weight,
            retain_weight=retain_weight,
        )

        if not torch.isfinite(loss):
            values = {name: value.detach().item()
                      for name, value in components.items()}
            raise FloatingPointError(
                f"Non-finite SALUN loss before backward: {values}"
            )

        loss.backward()

        # SALUN hard mask: only salient parameters receive an update.
        for name, parameter in net.named_parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(mask[name])

        grad_norm = torch.nn.utils.clip_grad_norm_(
            net.parameters(), max_grad_norm, error_if_nonfinite=True
        )
        optim.step()

        # Project non-salient parameters back to theta_0 and protect all model
        # buffers, including BatchNorm running statistics.
        with torch.no_grad():
            for name, parameter in net.named_parameters():
                parameter.copy_(
                    mask[name] * parameter
                    + (1.0 - mask[name]) * theta_0[name]
                )
            for name, buffer in net.named_buffers():
                buffer.copy_(buffers_0[name])

            for name, parameter in net.named_parameters():
                if not torch.isfinite(parameter).all():
                    raise FloatingPointError(
                        f"Parameter {name!r} became non-finite after update"
                    )

            generated_img = net.decoder(z_random)
            logits = _classifier_logits(identifier, generated_img)

        elapsed_time = time.time() - time_0
        metrics = {
            name: value.detach().item() for name, value in components.items()
        }
        metrics["total"] = loss.detach().item()
        metrics["grad_norm"] = grad_norm.detach().item()
        return metrics, generated_img, logits, elapsed_time

    return process_batch


def forget_loss_fn(model, batch, kl_weight=1.0):
    """Mean forget-set ELBO used only to construct the saliency mask."""
    real_img_forget, _ = batch
    real_img_forget = real_img_forget.reshape(
        real_img_forget.shape[0], -1
    ).to(model.device)

    # Use the model's ordinary stochastic VAE training loss, matching SALUN's
    # definition of saliency as the forget-data training-loss gradient.
    reconstructed_forget, mu_forget, logvar_forget = model(real_img_forget)
    rec_forget = vl.mean_reconstruction_loss(
        reconstructed_forget, real_img_forget
    )
    kl_forget = vl.mean_kl_div(mu_forget, logvar_forget)
    return rec_forget + kl_weight * kl_forget


def compute_salun_mask(
    model,
    forget_loader,
    forget_loss_fn=forget_loss_fn,
    sparsity=0.5,
    max_batches=None,
    kl_weight=1.0,
):
    """Compute a global hard SALUN mask from the original forget gradient.

    ``sparsity`` is the fraction of parameter elements frozen at their original
    value. For example, 0.5 selects approximately the largest 50% of absolute
    gradient entries for unlearning.
    """
    if not 0.0 <= sparsity < 1.0:
        raise ValueError("sparsity must satisfy 0 <= sparsity < 1")

    was_training = model.training
    model.eval()  # Do not contaminate BatchNorm buffers with forget-only data.
    model.zero_grad(set_to_none=True)

    examples_used = 0
    batches_used = 0
    for batch in forget_loader:
        batch_size = batch[0].shape[0]
        loss = forget_loss_fn(model, batch, kl_weight=kl_weight)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss while computing SALUN mask")

        # The loss is a batch mean. Weight it by batch size so the accumulated
        # gradient is an exact example mean even for a smaller final batch.
        (loss * batch_size).backward()
        examples_used += batch_size
        batches_used += 1

        if max_batches is not None and batches_used >= max_batches:
            break

    if examples_used == 0:
        raise ValueError("Cannot compute a SALUN mask from an empty loader")

    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(examples_used)

    scores = [
        parameter.grad.detach().abs().flatten()
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not scores:
        raise RuntimeError("No parameter gradients were produced for the mask")

    scores = torch.cat(scores)
    gamma = torch.quantile(scores, sparsity)
    mask = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.grad is not None:
            mask[name] = (parameter.grad.detach().abs() >= gamma).to(
                dtype=parameter.dtype
            )
        else:
            mask[name] = torch.zeros_like(parameter)

    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()
        _freeze_batchnorm_stats(model)
    else:
        model.eval()

    return mask, gamma


def _next_batch(loader, iterator):
    """Read indefinitely from a shuffled loader without caching its batches."""
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def train(
    model,
    folder,
    num_steps,
    batch_size,
    latent_dim=2,
    save_steps=None,
    collect_interval='epoch',
    log_interval=10,
    kl_weight=1.0,
    uniformity_weight=0.0,
    retain_weight=1.0,
    all_digits=list(range(10)),
    forget_digit=1,
    img_ext='jpg',
    classifier_path="../data/MNIST/classifiers/MNISTClassifier.pth",
    data_path='../../data/MNIST',
    learning_rate=1e-3,
    mask_sparsity=0.5,
    mask_max_batches=100,
    max_grad_norm=5.0,
    **viz_kwargs,
):
    """Run masked class unlearning for an unconditional MNIST VAE.

    The paper's conditional generation loss cannot be literal for this model
    because it has no concept-conditioning input. This implementation maps its
    misaligned concept to a randomly paired retain posterior, aligns decoder
    predictions with shared latent noise, and applies the ordinary retain VAE
    objective. Only weights selected by the forget-gradient mask can change.
    """
    (
        net,
        dataloader,
        optim,
        z_random,
        identifier,
        sample_dir,
        checkpoint_dir,
        epoch_length,
        epochs,
        num_steps,
        save_steps,
        collect_interval,
        log_interval,
        csv_file,
        device,
        grid_size,
    ) = vt.init(
        model,
        folder,
        num_steps,
        batch_size,
        latent_dim=latent_dim,
        save_steps=save_steps,
        collect_interval=collect_interval,
        log_interval=log_interval,
        kl_weight=kl_weight,
        uniformity_weight=uniformity_weight,
        orthogonality_weight=0.0,
        all_digits=all_digits,
        forget_digit=forget_digit,
        img_ext=img_ext,
        classifier_path=classifier_path,
        train_mode='orthogonal',
        data_path=data_path,
        learning_rate=learning_rate,
    )

    mask, gamma = compute_salun_mask(
        net,
        dataloader['forget'],
        forget_loss_fn=forget_loss_fn,
        sparsity=mask_sparsity,
        max_batches=mask_max_batches,
        kl_weight=kl_weight,
    )
    selected = sum(mask_value.sum().item() for mask_value in mask.values())
    total = sum(mask_value.numel() for mask_value in mask.values())
    tqdm.write(
        f"SALUN mask threshold={gamma.item():.6g}, "
        f"selected={selected / total:.2%}"
    )

    process_batch = get_processor(
        net,
        identifier,
        z_random,
        (kl_weight, retain_weight),
        optim,
        mask,
        max_grad_norm=max_grad_norm,
    )
    log_results = vo.get_logger(identifier, csv_file, log_interval)
    save = vt.get_saver(net, save_steps, checkpoint_dir, epoch_length)
    collect_samples = vt.get_collector(
        sample_dir, collect_interval, grid_size, img_ext
    )

    real_img, _ = next(iter(dataloader['original']))
    real_img = real_img.reshape(real_img.shape[0], -1).to(device)
    retain_iterator = iter(dataloader['retain'])
    forget_iterator = iter(dataloader['forget'])

    for global_step in tqdm(range(1, num_steps + 1), desc="Unlearning steps"):
        (img_retain, _), retain_iterator = _next_batch(
            dataloader['retain'], retain_iterator
        )
        (img_forget, _), forget_iterator = _next_batch(
            dataloader['forget'], forget_iterator
        )

        metrics, generated_img, logits, elapsed_time = process_batch(
            img_retain, img_forget
        )
        log_results(
            step=global_step,
            losses=[
                None,
                None,
                None,
                None,
                metrics["total"],
            ],
            elapsed_time=elapsed_time,
            real_img=real_img,
            generated_img=generated_img,
            logits=logits,
        )
        save(step=global_step)
        collect_samples(generated_img, step=global_step)

    viz_kwargs.update({"folder": folder})
    viz.summarize_training(**viz_kwargs)
