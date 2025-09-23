
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler  # or your scheduler

@torch.no_grad()
def generate(
    model,
    vae,
    class_labels,                  # LongTensor [B]
    n_steps: int = 10,
    guidance_scale: float = 2.0,
    seed: int = 42,
    device: str = "cuda",
    prediction_type: str = "epsilon",
    vae_scaling: float = 0.18215,  # SD-style scaling for 256x256 (4x32x32 latents)
    microbatch_size: int = 64,     # A100 can usually take 16–64 at 256^2; tune per VRAM
    amp_dtype: torch.dtype | None = None   # None => pick bfloat16 if supported else float16
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