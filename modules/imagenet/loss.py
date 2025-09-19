import torch
import dit

DIFFUSION_ = dit.load_diffusion(1000)

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