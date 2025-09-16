import os, sys, math
import torch
import numpy as np
import matplotlib.pyplot as plt
import importlib, inspect
from contextlib import nullcontext

HERE = os.path.dirname(os.path.abspath(__file__))               # .../forget/modules/imagenet
FAST_DIT = os.path.abspath(os.path.join(HERE, '..', 'fast-DiT'))# .../forget/modules/fast-DiT

if FAST_DIT not in sys.path:
    sys.path.insert(0, FAST_DIT)

from models import DiT_models
from diffusion import create_diffusion  # diffusion scheduler/factory
from download import find_model         # auto-download checkpoints
from diffusers.models import AutoencoderKL



def load_DiT(folder, device):
    """
    Loads a DiT-XL/2 model and an AutoencoderKL model from a folder.

    Parameters
    ----------
    folder : str
        The path to the folder containing the model checkpoints.
    device : torch.device
        The device on which to load the models.

    Returns
    -------
    model : DiT_models
        The loaded DiT-XL/2 model.
    vae : AutoencoderKL
        The loaded AutoencoderKL model.
    """
    image_size = 256
    latent = image_size // 8

    model = DiT_models["DiT-XL/2"](input_size=latent, num_classes=1000).to(device)
    state_dict = find_model(f"{folder}/DiT-XL-2-{image_size}x{image_size}.pt")
    model.load_state_dict(state_dict); model.eval()

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()
    return model, vae


# --- 1) Fix the diffusion object itself (preferred) ---
def fix_diffusion_fp32(diffusion):
    """
    Cast any float64 numpy arrays cached on the diffusion object to float32.
    Works for GaussianDiffusion/SpacedDiffusion variants.
    """
    likely = [
        "betas", "alphas", "alphas_cumprod", "alphas_cumprod_prev",
        "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
        "sqrt_recip_alphas_cumprod", "sqrt_recipm1_alphas_cumprod",
        "posterior_variance", "posterior_log_variance_clipped",
        "posterior_mean_coef1", "posterior_mean_coef2",
    ]
    for name in likely:
        if hasattr(diffusion, name):
            arr = getattr(diffusion, name)
            if isinstance(arr, np.ndarray) and arr.dtype != np.float32:
                setattr(diffusion, name, arr.astype(np.float32))

    # Generic sweep (handles repo variations)
    for k, v in list(diffusion.__dict__.items()):
        if isinstance(v, np.ndarray) and v.dtype != np.float32:
            setattr(diffusion, k, v.astype(np.float32))
        elif isinstance(v, (list, tuple)):
            changed = False
            new = []
            for item in v:
                if isinstance(item, np.ndarray) and item.dtype != np.float32:
                    new.append(item.astype(np.float32)); changed = True
                else:
                    new.append(item)
            if changed:
                setattr(diffusion, k, type(v)(new))
        elif isinstance(v, dict):
            changed = False
            new = {}
            for kk, vv in v.items():
                if isinstance(vv, np.ndarray) and vv.dtype != np.float32:
                    new[kk] = vv.astype(np.float32); changed = True
                else:
                    new[kk] = vv
            if changed:
                setattr(diffusion, k, new)
    return diffusion



# --- 2) Patch the exact gaussian_diffusion module used by THIS diffusion object ---
def patch_extract_into_tensor_fp32(diffusion):
    """
    Monkey-patch the module-global _extract_into_tensor to be fp32-safe on MPS.
    Locates the correct module via the diffusion instance.
    """
    base_mod = inspect.getmodule(diffusion).__name__.rsplit('.', 1)[0]  # e.g. "diffusion"
    gd = importlib.import_module(base_mod + ".gaussian_diffusion")

    if getattr(gd, "_extract_into_tensor_patched_for_fp32", False):
        return  # already patched

    def _extract_into_tensor_fp32(arr, timesteps, broadcast_shape):
        # Accept either numpy arrays (usual) or torch tensors (defensive)
        if isinstance(arr, np.ndarray):
            arr = arr.astype(np.float32, copy=False)
            res = torch.from_numpy(arr).to(timesteps.device)[timesteps]
        else:
            # Fallback if some impl passed a torch.Tensor
            res = torch.as_tensor(arr, device=timesteps.device, dtype=torch.float32)[timesteps]
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res

    gd._extract_into_tensor = _extract_into_tensor_fp32
    gd._extract_into_tensor_patched_for_fp32 = True


def load_diffusion(n_steps, downcast=True):
    """
    Loads a diffusion object from the diffusion factory with the specified number of steps.

    Parameters
    ----------
    n_steps : int
        The number of steps in the diffusion process.
    downcast : bool, optional
        If True, any float64 numpy arrays inside the diffusion object are cast to float32. Default is True.

    Returns
    -------
    diffusion : dict
        The loaded diffusion object.
    """
    diffusion = create_diffusion(f"{n_steps}")
    
    if downcast:
        fix_diffusion_fp32(diffusion)
        patch_extract_into_tensor_fp32(diffusion)

    return diffusion 




@torch.inference_mode()
def generate_cfg_fast(
    model, vae, diffusion, class_id, n_samples, cfg_scale, device, show=True,
    *,
    use_amp=True,            # mixed precision for model+VAE
    channels_last=True,      # better memory access on Ampere+
    progress=False          # disable progress bar for speed
):
    image_size  = 256
    latent_size = image_size // 8

    model.eval();
    if channels_last:
        model.to(memory_format=torch.channels_last)
        try: vae.to(memory_format=torch.channels_last)
        except: pass

   
    y_cond = torch.full((n_samples,), class_id, device=device, dtype=torch.long)
    y_null = torch.full((n_samples,), NULL_CLASS_ID, device=device, dtype=torch.long)
    y = torch.cat([y_cond, y_null], dim=0)                      # (2n,)


    # initial noise
    z = torch.randn(2*n_samples, 4, latent_size, latent_size, device=device)

    # AMP context (bf16 on A100/L4; fp16 otherwise)
    amp_ctx = nullcontext()
    if use_amp and torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
        amp_ctx = torch.cuda.amp.autocast(dtype=amp_dtype)



    with amp_ctx:
        latents = diffusion.p_sample_loop(
            model.forward_with_cfg, z.shape, z,
            clip_denoised=False,
            model_kwargs=dict(y=y, cfg_scale=cfg_scale),
            progress=progress,
            device=device,
            # steps=steps,  # uncomment if your p_sample_loop accepts it
        )
        sample = latents / LATENT_SCALE
        imgs = vae.decode(sample).sample[:n_samples]
        

    if not show:
        return imgs

    # Faster plotting: one CPU hop + simple loop
    rows = math.ceil(math.sqrt(n_samples))
    cols = math.ceil(n_samples / rows)
    imgs_vis = (imgs.detach().float().clamp(-1, 1) + 1) * 0.5
    imgs_np = imgs_vis.permute(0, 2, 3, 1).cpu().numpy()

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n_samples:
            ax.imshow(imgs_np[i])
    plt.tight_layout(); plt.show()

    return imgs




def generate_uncond(model, vae, diffusion, n_samples, device, show=True):
    """
    Unconditional generation with a class-conditioned DiT by feeding only the null label.

    Parameters
    ----------
    model : DiT_models
        The DiT model (e.g., DiT-XL/2) with num_classes=1000.
    vae : AutoencoderKL
        Decoder used to map latents -> images.
    diffusion : object
        Diffusion sampler with p_sample_loop(...).
    n_samples : int
        Number of images to generate.
    device : torch.device
        CUDA or CPU device.
    show : bool, optional
        If True, shows a grid of generated images.


    Returns
    -------
    imgs : torch.Tensor
        Generated images, shape (n_samples, 3, image_size, image_size), range ~[-1, 1].
    """
    image_size=256
    latent_size = image_size // 8

    # Use only null labels for unconditional sampling
    y = torch.full((n_samples,), 1000, device=device, dtype=torch.long) # null_id = 1000

    # Initial noise in latent space
    z = torch.randn(n_samples, 4, latent_size, latent_size, device=device)

    with torch.no_grad():
        latents = diffusion.p_sample_loop(
            model.forward,                               # no CFG path; plain forward
            (n_samples, 4, latent_size, latent_size),    # explicit shape
            z,
            clip_denoised=False,
            model_kwargs=dict(y=y),                      # supply null labels
            progress=True,
            device=device
        )
        samples = latents / 0.18215
        imgs = vae.decode(samples).sample  # (n_samples, 3, H, W), ~[-1, 1]

    if show:
        rows = math.ceil(math.sqrt(n_samples))
        cols = math.ceil(n_samples / rows)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes = np.atleast_1d(axes).ravel()

        for i in range(rows * cols):
            axes[i].axis("off")
            if i < n_samples:
                img_vis = imgs[i].detach().float().clamp(-1, 1)
                img_vis = (img_vis + 1) * 0.5
                img_np = img_vis.permute(1, 2, 0).cpu().numpy()
                axes[i].imshow(img_np)

        plt.tight_layout()
        plt.show()

    return imgs



def freeze_except_y_and_lastK_adaln(model, K=4, unfreeze_final_layer=False):
    """
    Freeze everything except:
      - y_embedder (always)
      - adaLN_modulation in the last K transformer blocks
      - (optional) final_layer if unfreeze_final_layer=True

    Returns:
      model, stats  (where stats is a dict with param counts)
    """
    # 1) Freeze all
    for p in model.parameters():
        p.requires_grad = False

    trainable_names = []

    # 2) y_embedder
    y_params = 0
    if hasattr(model, "y_embedder"):
        for n, p in model.y_embedder.named_parameters(prefix="y_embedder"):
            p.requires_grad = True
            y_params += p.numel()
            trainable_names.append(n)

    # 3) last-K adaLN_modulation
    adaln_params = 0
    num_blocks = len(getattr(model, "blocks", []))
    K = max(0, min(K, num_blocks))  # clamp
    if num_blocks and K > 0:
        for i, blk in enumerate(model.blocks):
            if i >= num_blocks - K and hasattr(blk, "adaLN_modulation"):
                for n, p in blk.adaLN_modulation.named_parameters(prefix=f"blocks.{i}.adaLN_modulation"):
                    p.requires_grad = True
                    adaln_params += p.numel()
                    trainable_names.append(n)

    # 4) optional final_layer
    final_params = 0
    if unfreeze_final_layer and hasattr(model, "final_layer"):
        for n, p in model.final_layer.named_parameters(prefix="final_layer"):
            p.requires_grad = True
            final_params += p.numel()
            trainable_names.append(n)

    # 5) counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    stats = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "breakdown": {
            "y_embedder": y_params,
            "adaLN_modulation_lastK": adaln_params,
            "final_layer": final_params,
        },
        "K": K,
        "num_blocks": num_blocks,
        "trainable_names": trainable_names,  # for quick inspection
    }

    # Nice printout
    def _mk(m): return f"{m/1e6:.2f}M"
    print(f"Total params:     {_mk(total_params)}")
    print(f"Trainable params: {_mk(trainable_params)}  "
          f"(y: {_mk(y_params)}, adaLN_last{K}: {_mk(adaln_params)}, final: {_mk(final_params)})")

    return model, stats





@torch.no_grad()
def encode_to_latents(
    vae,
    x,                          # (B,3,H,W) in [0,1] or [-1,1]
    image_size=256,
    vae_downsample_factor=8,    # 8 for SD/DiT VAEs
    vae_scaling_factor=0.18215, # scale used by SD AutoencoderKL; adjust if yours differs
    sample_posterior=False,      # True: sample; False: use mean
):


    # Normalize to [-1,1] for most VAEs
    x = x.clamp(-1., 1.)

    # Encode
    posterior = vae.encode(x)
    # diffusers AutoencoderKL: posterior has .latent_dist with .sample() / .mean
    # if hasattr(posterior, "latent_dist"):
    dist = posterior.latent_dist
    latents = dist.mean # dist.sample() if sample_posterior else 
    # else:
    #     # fallback if encode() directly returns a distribution
    #     latents = posterior.sample() if sample_posterior else posterior.mean

    # Scale to match training convention
    return latents * vae_scaling_factor
    










LATENT_SCALE = 0.18215
NULL_CLASS_ID = 1000

@torch.inference_mode()  # slightly faster than no_grad for inference
def generate_cfg_steady_fast(
    model, vae, diffusion, class_id, n_samples, cfg_scale, device,
    noise, show=True,
    *,
    fast_steps=None,          # e.g. 20–30 for previews (if your diffusion supports it)
    use_amp=True,             # fp16/bf16 decode + model forward
    use_mode_decode=True,     # deterministic & a little faster than .sample
    prebuilt_labels=None,     # pass cached labels if you’re calling this in a loop
    channels_last=True,       # enable channels_last for conv-heavy models
    progress=False            # progress bar costs a bit; disable during training
):
    """
    Fast preview sampler for training loops.
    """

    # ---------- caching / setup ----------
    model.eval(); 

    if channels_last:
        # This helps on Ampere+; harmless if already channels_last
        model.to(memory_format=torch.channels_last)
        # VAE path is mostly convs too
        for m in [vae]:
            try:
                m.to(memory_format=torch.channels_last)
            except Exception:
                pass

    # labels: [cond..., null...]
    if prebuilt_labels is not None:
        y = prebuilt_labels.to(device)
    else:
        y_cond = torch.full((n_samples,), class_id, device=device, dtype=torch.long)
        y_null = torch.full((n_samples,), NULL_CLASS_ID, device=device, dtype=torch.long)
        y = torch.cat([y_cond, y_null], dim=0)

    z = noise  # (2*n, 4, H/8, W/8) provided by caller (fixed noise)

    # ---------- AMP context (bf16 on A100/L4; fp16 elsewhere) ----------
    amp_ctx = nullcontext()
    if use_amp and torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
        amp_ctx = torch.cuda.amp.autocast(dtype=amp_dtype)

    # ---------- (optional) fewer sampling steps for previews ----------
    # If your `diffusion` object exposes something like `set_timesteps`,
    # use it to downsample the schedule for speed.
    if fast_steps is not None:
        try:
            diffusion.set_timesteps(fast_steps)  # HuggingFace-style API
        except Exception:
            # If not supported, ignore; some implementations accept a steps arg in p_sample_loop
            pass

    # ---------- sampling ----------
    with amp_ctx:
        latents = diffusion.p_sample_loop(
            model.forward_with_cfg,
            z.shape, z,
            clip_denoised=False,
            model_kwargs=dict(y=y, cfg_scale=cfg_scale),
            progress=progress,
            device=device,
            # steps=fast_steps,  # uncomment if your p_sample_loop supports it
        )

        # scale back and VAE decode
        sample = latents / LATENT_SCALE
        decoded = vae.decode(sample)
        imgs = decoded.mode() if use_mode_decode else decoded.sample
        imgs = imgs[:n_samples]  # keep only conditioned half

    if not show:
        return imgs

    return imgs





def generate_uncond_steady(model, vae, diffusion, n_samples, device, noise,show=True):
    """
    Unconditional generation with a class-conditioned DiT by feeding only the null label.

    Parameters
    ----------
    model : DiT_models
        The DiT model (e.g., DiT-XL/2) with num_classes=1000.
    vae : AutoencoderKL
        Decoder used to map latents -> images.
    diffusion : object
        Diffusion sampler with p_sample_loop(...).
    n_samples : int
        Number of images to generate.
    device : torch.device
        CUDA or CPU device.
    noise : torch.Tensor
        Initial noise in latent space of shape (n_samples, 4, latent_size, latent_size).
    show : bool, optional
        If True, shows a grid of generated images.

    Returns
    -------
    imgs : torch.Tensor
        Generated images, shape (n_samples, 3, image_size, image_size), range ~[-1, 1].
    """
    image_size=256
    latent_size = image_size // 8

    # Use only null labels for unconditional sampling
    y = torch.full((n_samples,), 1000, device=device, dtype=torch.long) # null_id = 1000

    # Initial noise in latent space
    z = noise # torch.randn(n_samples, 4, latent_size, latent_size, device=device)

    with torch.no_grad():
        latents = diffusion.p_sample_loop(
            model.forward,                               # no CFG path; plain forward
            (n_samples, 4, latent_size, latent_size),    # explicit shape
            z,
            clip_denoised=False,
            model_kwargs=dict(y=y),                      # supply null labels
            progress=True,
            device=device
        )
        samples = latents / 0.18215
        imgs = vae.decode(samples).sample  # (n_samples, 3, H, W), ~[-1, 1]

    if show:
        rows = math.ceil(math.sqrt(n_samples))
        cols = math.ceil(n_samples / rows)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes = np.atleast_1d(axes).ravel()

        for i in range(rows * cols):
            axes[i].axis("off")
            if i < n_samples:
                img_vis = imgs[i].detach().float().clamp(-1, 1)
                img_vis = (img_vis + 1) * 0.5
                img_np = img_vis.permute(1, 2, 0).cpu().numpy()
                axes[i].imshow(img_np)

        plt.tight_layout()
        plt.show()

    return imgs






def generate_cfg_batched(
    model, vae, diffusion,
    class_id: int,
    n_samples: int,
    cfg_scale: float,
    device: torch.device,
    batch_size: int = 8,
    show: bool = False,
):
    """
    Batched CFG sampling for DiT-XL/2 (+ AutoencoderKL) at 256x256.

    Returns
    -------
    imgs : Tensor, shape (n_samples, 3, 256, 256), in [-1, 1]
    """
    image_size = 256
    latent_size = image_size // 8  # 32 for 256x256
    out = []

    # choose autocast settings per device
    dev_type = str(device)  # "cuda", "mps", or "cpu"
    ac_dtype = torch.float16 if dev_type in ("cuda", "mps") else torch.bfloat16

    with torch.no_grad(), torch.amp.autocast(device_type=dev_type, dtype=ac_dtype, enabled=True):
        remaining = n_samples
        while remaining > 0:
            b = min(batch_size, remaining)

            # labels: [cond ... , null ...]  (2b,)
            y_cond = torch.full((b,), class_id, device=device, dtype=torch.long)
            y_null = torch.full((b,), 1000,    device=device, dtype=torch.long)  # null class
            y = torch.cat([y_cond, y_null], dim=0)

            # initial noise z: (2b, 4, 32, 32)
            z = torch.randn(2*b, 4, latent_size, latent_size, device=device)

            # reverse process
            latents = diffusion.p_sample_loop(
                model.forward_with_cfg, z.shape, z,
                clip_denoised=False,
                model_kwargs=dict(y=y, cfg_scale=cfg_scale),
                progress=False,
                device=device
            )

            # decode
            sample = latents / 0.18215
            imgs_b = vae.decode(sample).sample
            imgs_b = imgs_b[:b]                   # keep only the conditioned half
            out.append(imgs_b)

            remaining -= b

    imgs = torch.cat(out, dim=0)  # (n_samples, 3, 256, 256)

    return imgs.to(torch.float32)
