
import os, torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler 
import utility as ut 

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








@ut.timer
@torch.no_grad()
def gen_x(
    model,
    vae,
    x,                              # [B,4,32,32] latents already provided
    class_label,               # single int; labels built inside
    n_steps: int = 10,
    guidance_scale: float = 2.0,
    device: str = "mps",            # <- default to mps for Macs
    prediction_type: str = "epsilon",
    vae_scaling: float = 0.18215,
    microbatch_size: int = 16,      # <- smaller default for MPS VRAM
    amp_dtype: torch.dtype | None = None,
    cfg_concat: bool = False,       # <- False on MPS to reduce peak memory
    vae_decode_chunk: int = 8,      # <- decode in chunks on MPS
):
    """
    MPS-focused changes:
      - autocast(device_type='mps', dtype=torch.float16)
      - optional non-concat CFG to cut memory
      - chunked VAE decode to avoid OOM
      - avoid CUDA-only optimizations (TF32, channels_last, flash attn)
      - allow CPU fallback for unsupported ops
    """
    # 1) Enable CPU fallback for odd ops (harmless on CUDA/CPU too)
    # os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    

    dev = torch.device(device)
    x = x.to(dev, non_blocking=False)  # non_blocking is CUDA-only; no-op on MPS

    if amp_dtype is None:
        # MPS: prefer float16 (bf16 is inconsistent on Metal as of today)
        amp_dtype = torch.float16 if dev.type == "mps" else (
            torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        )

    # 2) Scheduler setup
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=1e-4,
        beta_end=2e-2,
        beta_schedule="linear",
        clip_sample=False,
    )
    scheduler.config.prediction_type = prediction_type
    scheduler.set_timesteps(n_steps, device=dev)
    timesteps = scheduler.timesteps

    B_total = x.shape[0]
    if isinstance(class_label, int):
        class_labels = torch.full((B_total,), class_label, dtype=torch.long, device=device)
    elif isinstance(class_label, list):
        class_labels = torch.tensor(class_label, dtype=torch.long, device=device)
    else:
        class_labels = class_label.to(device=device, dtype=torch.long)

    outs = []

    # 3) Microbatch loop (smaller for MPS)
    for start in range(0, B_total, microbatch_size):
        end = min(start + microbatch_size, B_total)
        bs = end - start

        labels_mb = class_labels[start:end]
        null_labels = torch.full_like(labels_mb, fill_value=1000)

        x_mb = x[start:end].to(dev, dtype=amp_dtype)

        # Pre-alloc timesteps
        t_batch = torch.empty(bs, dtype=torch.long, device=dev)
        if cfg_concat:
            t_batch2 = torch.empty(bs * 2, dtype=torch.long, device=dev)

        # 4) Diffusion loop
        #    Use autocast on MPS with fp16
        with torch.autocast(device_type=dev.type, dtype=amp_dtype):
            for t in timesteps:
                t_batch.fill_(t)

                if cfg_concat:
                    # (A) concat trick = 1 forward/step, higher memory
                    x_in = torch.cat([x_mb, x_mb], dim=0)
                    labels_cat = torch.cat([null_labels, labels_mb], dim=0)
                    t_batch2[:bs] = t_batch
                    t_batch2[bs:] = t_batch

                    eps = model(x_in, t_batch2, labels_cat)[:, :4]
                    eps_u, eps_c = eps[:bs], eps[bs:]
                else:
                    # (B) two passes = lower peak memory, good for MPS
                    eps_u = model(x_mb, t_batch, null_labels)[:, :4]
                    eps_c = model(x_mb, t_batch, labels_mb)[:, :4]

                eps_guided = eps_u + guidance_scale * (eps_c - eps_u)
                x_mb = scheduler.step(eps_guided, t, x_mb, eta=0.0).prev_sample

            # 5) Chunked VAE decode (helps avoid MPS OOM)
            z = x_mb / vae_scaling
            imgs_mb = []
            for i in range(0, bs, vae_decode_chunk):
                j = min(i + vae_decode_chunk, bs)
                # keep decode on MPS to stay on-GPU; if you see OOMs, do `.to("cpu")` before decode
                decoded = vae.decode(z[i:j]).sample
                decoded = (decoded.clamp(-1, 1) + 1) / 2
                imgs_mb.append(decoded)

            imgs_mb = torch.cat(imgs_mb, dim=0)

        outs.append(imgs_mb.to(dtype=torch.float32))

        # MPS tip: occasional sync helps benchmark accuracy (optional)
        if dev.type == "mps":
            torch.mps.synchronize()

        del x_mb, z, imgs_mb

    return torch.cat(outs, dim=0)
