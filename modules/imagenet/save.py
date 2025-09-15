import torch

def save_trainable_checkpoint(model, path, *, as_delta=False, base_model=None, dtype=torch.float32):
    """
    Saves only the parameters with requires_grad=True.
    If as_delta=True, stores (finetuned - base) so you can ship tiny 'diff' files.
    """
    trainable = {}
    base_sd = base_model.state_dict() if (as_delta and base_model is not None) else None

    for name, p in model.named_parameters():
        if p.requires_grad:
            t = p.detach().cpu().to(dtype)
            if as_delta:
                assert base_sd is not None, "Provide base_model when saving deltas."
                t = t - base_sd[name].detach().cpu().to(dtype)
            trainable[name] = t
    torch.save({"as_delta": as_delta, "state_dict": trainable, "dtype": str(dtype)}, path)

@torch.no_grad()
def apply_trainable_checkpoint(model, path, map_location="cpu"):
    """
    Loads a small checkpoint onto an already-loaded base model.
    Works with either absolute values or deltas.
    """
    ckpt = torch.load(path, map_location=map_location)
    is_delta = ckpt.get("as_delta", False)
    upd = ckpt["state_dict"]

    # quick name->param map
    name_to_param = dict(model.named_parameters())
    for name, t in upd.items():
        if name not in name_to_param:
            # if your finetune used DDP and added 'module.' prefix, strip it here, etc.
            continue
        p = name_to_param[name]
        T = t.to(p.dtype).to(p.device)
        if is_delta:
            p.add_(T)        # p = p + delta
        else:
            p.copy_(T)       # p = absolute finetuned value

# Example usage:
# 1) After finetune: save_trainable_checkpoint(model, "dit_xl2_patch.pt", as_delta=True, base_model=base_model)
# 2) Inference: load base DiT-XL/2, then apply_trainable_checkpoint(model, "dit_xl2_patch.pt")
