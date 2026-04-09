import torch
import dit

DIFFUSION_ = dit.load_diffusion(1000)
sqrt_alphas_cumprod = torch.tensor(DIFFUSION_.sqrt_alphas_cumprod, device="cuda", dtype=torch.float32)
sqrt_one_minus_alphas_cumprod = torch.tensor(DIFFUSION_.sqrt_one_minus_alphas_cumprod, device="cuda", dtype=torch.float32)

def loss(
    model,
    vae,
    diffusion,
    device,
    x,                          # (B,C,H,W) normalized as per diffusion
    y=None,                     # (B,) labels or None for unconditional
    cfg_drop_prob=0.1,          # classifier-free guidance dropout prob
    null_label=1000,            # index of "null" class in embedding table
):
 

    with torch.no_grad():
        # Map input images to latent space + normalize latents:
        x = vae.encode(x).latent_dist.sample().mul_(0.18215)

    B = x.shape[0]
    # Guard against numpy.int64, etc.
    num_steps = int(getattr(DIFFUSION_, "num_timesteps"))
    t = torch.randint(
        low=0,
        high=num_steps,
        size=(B,),
        device=device,
    )

    # Classifier-free guidance dropout (only if conditional)
    # y_tilde = y.to(device=device)
    # drop = torch.rand(B, device=device) < cfg_drop_prob
    # y_tilde = y_tilde.masked_fill(drop, null_label)  # safe boolean masking
    model_kwargs = {"y": y}#y_tilde}

    losses = DIFFUSION_.training_losses(
        model=model,
        x_start=x,
        t=t,
        model_kwargs=model_kwargs,
    )

    # per_example = losses["mse"] if "mse" in losses else losses["loss"]
    # if weight_fn is not None:
    #     per_example = per_example * weight_fn(t).to(per_example)
    return losses["loss"].mean()




def loss_pure(model, vae, device, img, label):
    t = torch.randint(0, 1000, (img.shape[0],), device=device, dtype=torch.long)

    with torch.no_grad():
        z = vae.encode(img).latent_dist.sample().mul_(0.18215)
        
    noise = torch.randn_like(z)

    a = sqrt_alphas_cumprod[t].view(img.shape[0], 1, 1, 1)
    b = sqrt_one_minus_alphas_cumprod[t].view(img.shape[0], 1, 1, 1)
    # x_t = a * z + b * noise

    return torch.mean((model(a * z + b * noise, t, label) - noise) ** 2)