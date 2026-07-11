"""Saliency Unlearning (SalUn) for conditional ImageNet DiT models.

This module implements Algorithm 2 and Eq. (7) from:

    SalUn: Empowering Machine Unlearning via Gradient-based Weight Saliency
    in Both Image Classification and Generation (ICLR 2024).

It intentionally reuses the ImageNet experiment setup, data loaders, model
loading, sampling logger, checkpoint format, and visualizations from train.py.
"""

from contextlib import nullcontext
import json
import os
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

import save as sv
import train as tt
import viz


LATENT_SCALE = 0.18215


def _amp_config(device, use_amp):
    device = torch.device(device)
    enabled = bool(
        use_amp and device.type == "cuda" and torch.cuda.is_available()
    )
    if not enabled:
        return False, None
    if torch.cuda.is_bf16_supported():
        return True, torch.bfloat16
    return True, torch.float16


def _autocast(device, enabled, dtype):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=torch.device(device).type, dtype=dtype)


def _grad_scaler(enabled):
    return torch.amp.GradScaler("cuda", enabled=enabled)


@torch.no_grad()
def _encode_images(vae, images):
    """Encode with the frozen VAE in FP32, independent of DiT autocast."""
    with torch.autocast(device_type=images.device.type, enabled=False):
        latents = vae.encode(images.float()).latent_dist.sample()
    return latents.float().mul_(LATENT_SCALE)


def _wrapped_model(diffusion, model):
    """Map reduced diffusion indices back to DiT's original 1000-step indices."""
    wrap = getattr(diffusion, "_wrap_model", None)
    return wrap(model) if wrap is not None else model


def _epsilon_prediction(wrapped_model, x_t, timesteps, labels):
    """Return only epsilon channels, excluding DiT's learned-variance output."""
    output = wrapped_model(x_t, timesteps, y=labels)
    if isinstance(output, (tuple, list)):
        output = output[0]
    latent_channels = x_t.shape[1]
    if output.shape[1] < latent_channels:
        raise RuntimeError(
            "DiT output has fewer channels than its noised latent input"
        )
    return output[:, :latent_channels]


def _diffusion_inputs(diffusion, latents):
    """Draw t and epsilon once and construct x_t from the supplied latents."""
    batch_size = latents.shape[0]
    timesteps = torch.randint(
        0,
        int(diffusion.num_timesteps),
        (batch_size,),
        device=latents.device,
        dtype=torch.long,
    )
    noise = torch.randn_like(latents)
    x_t = diffusion.q_sample(latents, timesteps, noise=noise)
    return x_t, timesteps, noise


def diffusion_mse_loss(model, vae, diffusion, images, labels):
    """The standard conditional epsilon-prediction loss l_MSE(theta; D)."""
    latents = _encode_images(vae, images)
    x_t, timesteps, noise = _diffusion_inputs(diffusion, latents)
    epsilon = _epsilon_prediction(
        _wrapped_model(diffusion, model), x_t, timesteps, labels
    )
    return F.mse_loss(epsilon.float(), noise.float())


def salun_generation_loss(
    model,
    vae,
    diffusion,
    forget_images,
    forget_labels,
    alternative_labels,
    retain_images,
    retain_labels,
    retain_weight=1.0,
):
    """Eq. (7): misaligned-condition MSE plus beta times retain MSE.

    The same forgotten x_t and timestep are evaluated under c and c', exactly
    as in the paper. Both predictions come from the current unlearned model;
    neither branch is detached and no classifier or teacher participates.
    """
    wrapped = _wrapped_model(diffusion, model)

    forget_latents = _encode_images(vae, forget_images)
    forget_x_t, forget_t, _ = _diffusion_inputs(diffusion, forget_latents)
    epsilon_original = _epsilon_prediction(
        wrapped, forget_x_t, forget_t, forget_labels
    )
    epsilon_alternative = _epsilon_prediction(
        wrapped, forget_x_t, forget_t, alternative_labels
    )
    forget_loss = F.mse_loss(
        epsilon_alternative.float(), epsilon_original.float()
    )

    retain_latents = _encode_images(vae, retain_images)
    retain_x_t, retain_t, retain_noise = _diffusion_inputs(
        diffusion, retain_latents
    )
    epsilon_retain = _epsilon_prediction(
        wrapped, retain_x_t, retain_t, retain_labels
    )
    retain_loss = F.mse_loss(epsilon_retain.float(), retain_noise.float())

    total_loss = forget_loss + retain_weight * retain_loss
    return total_loss, forget_loss, retain_loss


def compute_salun_mask(
    model,
    vae,
    diffusion,
    forget_loader,
    optimizer,
    device,
    sparsity=0.5,
    max_batches=None,
    use_amp=True,
):
    """Compute m_S = 1(|grad l_MSE(theta_o; D_f)| >= gamma).

    The threshold is global over parameters currently marked trainable by the
    ImageNet model setup. Pass ``keep_all=True`` to ``train`` to reproduce the
    paper's full-model mask; the repository's selective DiT configuration is
    supported for lower-memory experiments.
    """
    if not 0.0 <= sparsity < 1.0:
        raise ValueError("sparsity must satisfy 0 <= sparsity < 1")
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive or None")

    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("The DiT model has no trainable parameters")

    amp_enabled, amp_dtype = _amp_config(device, use_amp)
    scaler = _grad_scaler(amp_enabled and amp_dtype == torch.float16)

    # Evaluation mode prevents classifier-free label dropout. Gradients remain
    # enabled, so this still computes the forget training-loss saliency.
    model.eval()
    optimizer.zero_grad(set_to_none=True)
    examples_used = 0
    batches_used = 0

    for images, labels in tqdm(forget_loader, desc="SALUN mask"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True, dtype=torch.long)
        with _autocast(device, amp_enabled, amp_dtype):
            loss = diffusion_mse_loss(model, vae, diffusion, images, labels)

        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite forget loss during mask creation")

        weighted_loss = loss * images.shape[0]
        if scaler.is_enabled():
            scaler.scale(weighted_loss).backward()
        else:
            weighted_loss.backward()

        examples_used += images.shape[0]
        batches_used += 1
        if max_batches is not None and batches_used >= max_batches:
            break

    if examples_used == 0:
        raise ValueError("Cannot compute a SALUN mask from an empty loader")
    if scaler.is_enabled():
        scaler.unscale_(optimizer)

    # Move scores to CPU before concatenation to keep GPU peak memory lower.
    named_scores = []
    for name, parameter in trainable:
        if parameter.grad is None:
            score = torch.zeros_like(parameter, device="cpu", dtype=torch.float32)
        else:
            score = (
                parameter.grad.detach().float().abs().div_(examples_used).cpu()
            )
        named_scores.append((name, parameter, score))

    all_scores = torch.cat([score.flatten() for _, _, score in named_scores])
    keep_count = max(
        1, min(all_scores.numel(), round((1.0 - sparsity) * all_scores.numel()))
    )
    # Exact global top-k avoids gamma=0 selecting every class-embedding row
    # when many labels have zero forget gradient.
    kth_smallest = all_scores.numel() - keep_count + 1
    gamma = torch.kthvalue(all_scores, kth_smallest).values
    flat_mask = all_scores > gamma
    remaining = keep_count - int(flat_mask.sum().item())
    if remaining:
        # Select only as many gamma-tied entries as needed. This preserves an
        # exact global keep fraction without allocating top-k index tensors.
        offset = 0
        for _, _, score in named_scores:
            size = score.numel()
            chunk = flat_mask[offset:offset + size]
            tied = score.flatten() == gamma
            tied_count = int(tied.sum().item())
            take = min(remaining, tied_count)
            if take == tied_count:
                chunk |= tied
            elif take:
                tied_indices = torch.nonzero(tied, as_tuple=False)[:take, 0]
                chunk[tied_indices] = True
            remaining -= take
            offset += size
            if remaining == 0:
                break

    mask = {}
    offset = 0
    for name, parameter, score in named_scores:
        size = score.numel()
        mask[name] = flat_mask[offset:offset + size].reshape(score.shape).to(
            device=parameter.device
        )
        offset += size
    optimizer.zero_grad(set_to_none=True)

    selected = sum(value.sum().item() for value in mask.values())
    total = sum(value.numel() for value in mask.values())
    return mask, gamma.item(), selected / total


def _sample_alternative_labels(forget_labels, alternative_classes):
    choices = torch.as_tensor(
        alternative_classes,
        device=forget_labels.device,
        dtype=torch.long,
    )
    if choices.numel() == 0:
        raise ValueError("At least one alternative class is required")
    if torch.isin(forget_labels, choices).any():
        raise ValueError("Alternative classes must exclude the forget class")
    indices = torch.randint(
        choices.numel(), (forget_labels.shape[0],), device=forget_labels.device
    )
    return choices[indices]


def get_processor(
    model,
    vae,
    diffusion,
    optimizer,
    mask,
    device,
    alternative_classes,
    retain_weight=1.0,
    max_grad_norm=None,
    use_amp=True,
):
    """Create one Algorithm-2 masked SGD/AdamW update."""
    amp_enabled, amp_dtype = _amp_config(device, use_amp)
    scaler = _grad_scaler(amp_enabled and amp_dtype == torch.float16)
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]

    def process_batch(retain_images, retain_labels, forget_images, forget_labels):
        start_time = time.time()
        model.eval()  # Keep c and c' from being replaced by the null condition.
        optimizer.zero_grad(set_to_none=True)

        retain_images = retain_images.to(device, non_blocking=True)
        retain_labels = retain_labels.to(
            device, non_blocking=True, dtype=torch.long
        )
        forget_images = forget_images.to(device, non_blocking=True)
        forget_labels = forget_labels.to(
            device, non_blocking=True, dtype=torch.long
        )
        alternative_labels = _sample_alternative_labels(
            forget_labels, alternative_classes
        )

        with _autocast(device, amp_enabled, amp_dtype):
            total_loss, forget_loss, retain_loss = salun_generation_loss(
                model,
                vae,
                diffusion,
                forget_images,
                forget_labels,
                alternative_labels,
                retain_images,
                retain_labels,
                retain_weight=retain_weight,
            )

        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                "Non-finite SALUN objective: "
                f"forget={forget_loss.detach().item()}, "
                f"retain={retain_loss.detach().item()}"
            )

        if scaler.is_enabled():
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
        else:
            total_loss.backward()

        # Algorithm 2: theta_u <- theta_u - eta * (m_S elementwise-multiplied by g).
        for name, parameter in trainable:
            if parameter.grad is not None:
                parameter.grad.mul_(mask[name])

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                [parameter for _, parameter in trainable],
                max_grad_norm,
                error_if_nonfinite=True,
            )

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        elapsed_time = time.time() - start_time
        return (
            total_loss.detach().item(),
            forget_loss.detach().item(),
            retain_loss.detach().item(),
            elapsed_time,
        )

    return process_batch


def _next_batch(loader, iterator):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _record_salun_config(
    folder,
    retain_weight,
    mask_sparsity,
    mask_max_batches,
    alternative_classes,
    mask_use_amp,
    train_use_amp,
    mask_seed,
    keep_all,
):
    """Augment train.py's config with the SALUN-specific hyperparameters."""
    path = os.path.join(folder, "config.json")
    try:
        with open(path, "r") as file:
            config = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        config = {}

    training = config.setdefault("training", {})
    experiment = config.setdefault("experiment", {})
    training["salun_retain_weight"] = {
        "value": retain_weight,
        "description": "Beta multiplying the retained diffusion MSE.",
    }
    training["salun_mask_sparsity"] = {
        "value": mask_sparsity,
        "description": "Fraction of configured trainable weights masked out.",
    }
    training["salun_mask_max_batches"] = {
        "value": mask_max_batches,
        "description": "Forget batches used for the fixed saliency mask.",
    }
    training["salun_mask_use_amp"] = {
        "value": bool(mask_use_amp),
        "description": "Whether mask gradients use mixed precision on CUDA.",
    }
    training["salun_train_use_amp"] = {
        "value": bool(train_use_amp),
        "description": "Whether unlearning updates use mixed precision on CUDA.",
    }
    training["salun_mask_seed"] = {
        "value": mask_seed,
        "description": "Isolated RNG seed used while constructing the mask.",
    }
    experiment["salun_alternative_classes"] = {
        "value": list(alternative_classes),
        "description": "Misaligned c-prime classes used for forget examples.",
    }
    experiment["salun_full_model_mask"] = {
        "value": bool(keep_all),
        "description": "Whether saliency is computed over the full DiT model.",
    }
    with open(path, "w") as file:
        json.dump(config, file, indent=4)


def train(
    model_path,
    folder,
    num_steps,
    batch_size,
    save_steps=None,
    collect_interval="epoch",
    log_interval=10,
    learning_rate=1e-4,
    retain_weight=1.0,
    mask_sparsity=0.5,
    mask_max_batches=None,
    alternative_classes=None,
    exchange_classes=(208,),
    forget_class=207,
    img_ext="jpg",
    data_path="../../data/ImageNet-1k/2012",
    imagenet_json_path="../../data/ImageNet-1k/imagenet_1k.json",
    n_samples=100,
    device="cuda",
    diffusion_steps=1000,
    freeze_K=4,
    unfreeze_last=False,
    unfreeze_x_embedder=False,
    keep_all=False,
    mask_use_amp=False,
    train_use_amp=True,
    mask_seed=0,
    use_amp=None,
    max_grad_norm=None,
    save_mask=True,
    **gen_kwargs,
):
    """Run paper-style SalUn on ImageNet DiT-XL/2.

    ``exchange_classes`` supplies D_r in this repository's experiment setup.
    ``alternative_classes`` supplies c'; by default the same retained classes
    are used. For the paper's full-model saliency domain, set ``keep_all=True``;
    the default preserves the repository's memory-efficient DiT interaction.
    ``diffusion_steps`` defaults to the full 1000-step training schedule; sample
    generation remains controlled independently by ``n_steps`` in gen_kwargs.
    ``use_amp`` is a backwards-compatible alias that, when explicitly passed,
    overrides both ``mask_use_amp`` and ``train_use_amp``.
    """
    if retain_weight < 0:
        raise ValueError("retain_weight (beta) must be non-negative")
    if not exchange_classes:
        raise ValueError("exchange_classes must provide retained data")
    if not 0 <= int(forget_class) < 1000:
        raise ValueError("forget_class must be an ImageNet class in [0, 999]")
    if alternative_classes is None:
        alternative_classes = tuple(exchange_classes)
    alternative_classes = tuple(
        int(class_id)
        for class_id in alternative_classes
        if int(class_id) != int(forget_class)
    )
    if not alternative_classes:
        raise ValueError("alternative_classes must contain a non-forget class")
    if any(class_id < 0 or class_id >= 1000 for class_id in alternative_classes):
        raise ValueError("alternative_classes must be ImageNet classes in [0, 999]")
    if use_amp is not None:
        mask_use_amp = bool(use_amp)
        train_use_amp = bool(use_amp)

    gen_kwargs.setdefault("n_steps", 10)
    gen_kwargs.setdefault("guidance_scale", 2.0)

    (
        model,
        vae,
        diffusion,
        dataloader,
        optimizer,
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
        trainable_params,
    ) = tt.init(
        model_path,
        folder,
        num_steps,
        batch_size,
        save_steps=save_steps,
        collect_interval=collect_interval,
        log_interval=log_interval,
        learning_rate=learning_rate,
        uniformity_weight=0.0,
        orthogonality_weight=0.0,
        forget_weight=None,
        exchange_classes=list(exchange_classes),
        forget_class=forget_class,
        img_ext=img_ext,
        train_mode="orthogonal",
        data_path=data_path,
        imagenet_json_path=imagenet_json_path,
        n_samples=n_samples,
        device=device,
        diffusion_steps=diffusion_steps,
        freeze_K=freeze_K,
        unfreeze_last=unfreeze_last,
        unfreeze_x_embedder=unfreeze_x_embedder,
        keep_all=keep_all,
    )
    del trainable_params  # mask construction derives names directly from model.

    _record_salun_config(
        folder,
        retain_weight,
        mask_sparsity,
        mask_max_batches,
        alternative_classes,
        mask_use_amp,
        train_use_amp,
        mask_seed,
        keep_all,
    )

    # DiT uses activation checkpointing; non-reentrant mode supports the mask
    # and unlearning backward passes used here.
    tt.patch_checkpoint_nonreentrant()

    mask_device = torch.device(device)
    cuda_devices = []
    if mask_device.type == "cuda":
        cuda_devices = [
            mask_device.index
            if mask_device.index is not None
            else torch.cuda.current_device()
        ]
    rng_context = (
        torch.random.fork_rng(devices=cuda_devices)
        if mask_seed is not None
        else nullcontext()
    )
    with rng_context:
        if mask_seed is not None:
            torch.manual_seed(int(mask_seed))
            if mask_device.type == "cuda":
                torch.cuda.manual_seed(int(mask_seed))
        mask, gamma, selected_fraction = compute_salun_mask(
            model,
            vae,
            diffusion,
            dataloader["forget"],
            optimizer,
            device,
            sparsity=mask_sparsity,
            max_batches=mask_max_batches,
            use_amp=mask_use_amp,
        )
    tqdm.write(
        f"SALUN mask threshold={gamma:.6g}, "
        f"selected={selected_fraction:.2%}"
    )
    if save_mask:
        torch.save(
            {
                "mask": {name: value.cpu() for name, value in mask.items()},
                "threshold": gamma,
                "selected_fraction": selected_fraction,
                "sparsity": mask_sparsity,
            },
            os.path.join(checkpoint_dir, "salun_mask.pt"),
        )

    process_batch = get_processor(
        model,
        vae,
        diffusion,
        optimizer,
        mask,
        device,
        alternative_classes,
        retain_weight=retain_weight,
        max_grad_norm=max_grad_norm,
        use_amp=train_use_amp,
    )
    log_results = tt.get_logger(
        model,
        vae,
        diffusion,
        identifier,
        csv_file,
        log_interval,
        forget_class,
        z_random,
        **gen_kwargs,
    )
    save = tt.get_saver(model, save_steps, checkpoint_dir, epoch_length)
    collect_samples = tt.get_collector(
        sample_dir, collect_interval, grid_size, identifier, img_ext
    )

    retain_iterator = iter(dataloader["retain"])
    forget_iterator = iter(dataloader["forget"])

    for global_step in tqdm(range(1, num_steps + 1), desc="SALUN updates"):
        (retain_images, retain_labels), retain_iterator = _next_batch(
            dataloader["retain"], retain_iterator
        )
        (forget_images, forget_labels), forget_iterator = _next_batch(
            dataloader["forget"], forget_iterator
        )

        total_loss, _, _, elapsed_time = process_batch(
            retain_images,
            retain_labels,
            forget_images,
            forget_labels,
        )
        # Keep the repository's existing one-loss CSV and generation metrics.
        generated_img = log_results(
            step=global_step,
            losses=[total_loss],
            elapsed_time=elapsed_time,
        )
        save(step=global_step)
        collect_samples(generated_img, step=global_step)

        tt.free_gpu_memory(device)

    sv.save_trainable_checkpoint(
        model, os.path.join(checkpoint_dir, f"DiT_step_{num_steps}.pth")
    )
    viz.summarize_training(folder)
