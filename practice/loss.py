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
 

    # Ensure fp32 tensors on the target device
    # x = x.to(device=device, dtype=torch.float32, non_blocking=True)
    z = dit.encode_to_latents(vae, x)
    # noise = noise.to(device=device, dtype=torch.float32, non_blocking=True)

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
        x_start=z,
        t=t,
        model_kwargs=model_kwargs,
    )

    # per_example = losses["mse"] if "mse" in losses else losses["loss"]
    # if weight_fn is not None:
    #     per_example = per_example * weight_fn(t).to(per_example)
    return losses["loss"].mean()