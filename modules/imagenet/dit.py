import os, sys, math
import torch
import numpy as np
import matplotlib.pyplot as plt
import importlib, inspect
from contextlib import nullcontext
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

HERE = os.path.dirname(os.path.abspath(__file__))               # .../forget/modules/imagenet
FAST_DIT = os.path.abspath(os.path.join(HERE, '..', 'fast-DiT'))# .../forget/modules/fast-DiT

if FAST_DIT not in sys.path:
    sys.path.insert(0, FAST_DIT)

from models import DiT_models
from diffusion import create_diffusion  # diffusion scheduler/factory
from download import find_model         # auto-download checkpoints
from diffusers.models import AutoencoderKL
import gc


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




@torch.no_grad()
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







def make_generator(device: str, seed: int) -> torch.Generator:
    dev_type = "cuda" if device.startswith("cuda") else ("mps" if device.startswith("mps") else "cpu")
    return torch.Generator(device=dev_type).manual_seed(int(seed))



def freeze_except_y_and_lastK_adaln(
    model,
    K=4,
    unfreeze_final_layer=False,
    unfreeze_x_embedder=False,
    keep_all=False,
):
    """
    Configure trainable parameters.

    If keep_all=True:
        - Keep EVERYTHING trainable.
        - Still compute and print correct stats/breakdown.

    Else (default behavior):
        - Freeze everything except:
            * y_embedder (always)
            * adaLN_modulation in the last K transformer blocks
            * (optional) x_embedder if unfreeze_x_embedder=True
            * (optional) final_layer if unfreeze_final_layer=True

    Returns:
        model, stats  (stats is a dict with param counts and breakdown)
    """
    # --- Common meta ---
    num_blocks = len(getattr(model, "blocks", []))
    K = max(0, min(K, num_blocks))  # clamp to available blocks

    # Helpers
    def _sum_params(module):
        return sum(p.numel() for p in module.parameters()) if module is not None else 0

    def _named_params(module, prefix):
        if module is None:
            return []
        return [f"{prefix}.{n}" for n, _ in module.named_parameters()]

    # Initialize counters/containers
    trainable_names = []
    x_params = y_params = adaln_params = final_params = 0

    if keep_all:
        # -------- Keep EVERYTHING trainable, but still compute proper stats --------
        for p in model.parameters():
            p.requires_grad = True

        # Breakdown counts (informational only)
        x_params = _sum_params(getattr(model, "x_embedder", None))
        y_params = _sum_params(getattr(model, "y_embedder", None))
        if hasattr(model, "final_layer"):
            final_params = _sum_params(model.final_layer)

        if num_blocks and K > 0:
            for i, blk in enumerate(model.blocks):
                if i >= num_blocks - K and hasattr(blk, "adaLN_modulation"):
                    adaln_params += _sum_params(blk.adaLN_modulation)

        # Names (for quick inspection). Listing all can be long; keep it consistent:
        trainable_names = [n for n, _ in model.named_parameters()]

    else:
        # -------- Original selective unfreeze path --------
        # 1) Freeze all
        for p in model.parameters():
            p.requires_grad = False

        # 2) y_embedder (always)
        if hasattr(model, "y_embedder"):
            for n, p in model.y_embedder.named_parameters(prefix="y_embedder"):
                p.requires_grad = True
                y_params += p.numel()
                trainable_names.append(n)

        # 2b) optional x_embedder
        if unfreeze_x_embedder and hasattr(model, "x_embedder"):
            for n, p in model.x_embedder.named_parameters(prefix="x_embedder"):
                p.requires_grad = True
                x_params += p.numel()
                trainable_names.append(n)

        # 3) last-K adaLN_modulation
        if num_blocks and K > 0:
            for i, blk in enumerate(model.blocks):
                if i >= num_blocks - K and hasattr(blk, "adaLN_modulation"):
                    for n, p in blk.adaLN_modulation.named_parameters(prefix=f"blocks.{i}.adaLN_modulation"):
                        p.requires_grad = True
                        adaln_params += p.numel()
                        trainable_names.append(n)

        # 4) optional final_layer
        if unfreeze_final_layer and hasattr(model, "final_layer"):
            for n, p in model.final_layer.named_parameters(prefix="final_layer"):
                p.requires_grad = True
                final_params += p.numel()
                trainable_names.append(n)

    # --- Stats / printout (works for both branches) ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    stats = {
        "keep_all": bool(keep_all),
        "unfreeze_x_embedder": bool(unfreeze_x_embedder),
        "unfreeze_final_layer": bool(unfreeze_final_layer),
        "K": K,
        "num_blocks": num_blocks,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "breakdown": {
            "x_embedder": x_params,
            "y_embedder": y_params,
            "adaLN_modulation_lastK": adaln_params,
            "final_layer": final_params,
        },
        "trainable_names": trainable_names,  # for quick inspection
    }

    def _mk(m): return f"{m/1e6:.2f}M"
    print(f"Total params:     {_mk(total_params)}")
    print(
        f"Trainable params: {_mk(trainable_params)}  "
        f"(x: {_mk(x_params)}, y: {_mk(y_params)}, "
        f"adaLN_last{K}: {_mk(adaln_params)}, final: {_mk(final_params)})"
    )

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

@torch.no_grad()
def generate(
    model,
    vae,
    class_labels,                  # LongTensor [B]
    n_steps: int = 50,
    guidance_scale: float = 8.0,
    seed: int = 42,
    device: str = "cuda",
    prediction_type: str = "epsilon",
    vae_scaling: float = 0.18215,  # SD-style scaling for 256x256 (4x32x32 latents)
    microbatch_size: int = 64,     # A100 can usually take 16–64 at 256^2; tune per VRAM
    amp_dtype: torch.dtype | None = None,
    **kwargs  # None => pick bfloat16 if supported else float16
):
    """
    Optimized sampler for A100:
      - autocast (bf16 on A100 by default)
      - TF32 matmul enabled
      - Flash/SDP attention (PyTorch 2.x)
      - channels_last
      - classifier-free guidance via single forward per step (concat trick)
      - VAE decode under autocast on GPU
    Returns: float32 images in [0,1], shape [B,3,H,W].
    """

    B_total = class_labels.shape[0]
    device = torch.device(device)
    # Choose AMP dtype: A100 supports bf16 well; fall back to fp16
    if amp_dtype is None:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # ---- Scheduler ----
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=1e-4,
        beta_end=2e-2,
        beta_schedule="linear",
        clip_sample=False,
    )
    scheduler.config.prediction_type = prediction_type
    scheduler.set_timesteps(n_steps, device=device)
    timesteps = scheduler.timesteps

    # Deterministic seeds per-sample so results are independent of microbatching
    base_gen = torch.Generator(device=device).manual_seed(seed)
    outs = []

    # Process in micro-batches
    for start in range(0, B_total, microbatch_size):
        end = min(start + microbatch_size, B_total)
        bs  = end - start
        labels_mb = class_labels[start:end].to(device, non_blocking=True)
        null_labels = torch.full_like(labels_mb, fill_value=1000)

        # fixed per-sample seeds
        gen = torch.Generator(device=device).manual_seed(seed + start)

        # Init latent noise (4x32x32 for 256x256 with SD-style VAE)
        x = torch.randn(bs, 4, 32, 32, generator=gen, device=device, dtype=amp_dtype)

        # Pre-allocate timestep tensor (+ doubled for CFG concat)
        t_batch = torch.empty(bs, dtype=torch.long, device=device)
        t_batch2 = torch.empty(bs * 2, dtype=torch.long, device=device)

        # Diffusion loop with CFG "concat trick": one forward per step
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            for t in timesteps:
                t_batch.fill_(t)
                # concat [uncond, cond]
                x_in = torch.cat([x, x], dim=0)
                labels_cat = torch.cat([null_labels, labels_mb], dim=0)
                t_batch2[:bs] = t_batch; t_batch2[bs:] = t_batch

                # Model forward once
                eps = model(x_in, t_batch2, labels_cat)
                # if isinstance(eps, (tuple, list)):
                # eps = eps[0]
                # if eps.shape[1] > 4:
                eps = eps[:, :4]  # drop variance channels if present

                # Split and combine
                eps_u, eps_c = eps[:bs], eps[bs:]
                eps_guided = eps_u + guidance_scale * (eps_c - eps_u)

                # DDIM update (eta=0 deterministic)
                x = scheduler.step(eps_guided, t, x, eta=0.0).prev_sample

            # Decode latents -> [0,1]
            z = x / vae_scaling
            imgs = vae.decode(z).sample[:B_total] 
            imgs = (imgs.clamp(-1, 1) + 1) / 2

        # Cast to float32 for downstream code
        outs.append(imgs.to(dtype=torch.float32))

        # housekeeping
        del x, z, imgs

    return torch.cat(outs, dim=0)





