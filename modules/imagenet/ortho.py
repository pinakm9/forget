import train as tt 
import torch, time, csv
import loss as ls
from torch.autograd import grad
from tqdm import tqdm
import viz
import torch.utils.checkpoint as _cp
import classifier as cl
import dit
from torch.cuda.amp import autocast, GradScaler
import gc


def get_processor(model, vae, diffusion, device, optim, trainable_params, orthogonality_weight):
    device = vae.device
    cap_major = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
    AMP_DTYPE = torch.bfloat16 if cap_major >= 8 else torch.float16   # A100/L4 → bf16, T4/V100 → fp16
    AMP_ENABLED = torch.cuda.is_available()
    scaler = GradScaler(enabled=(AMP_ENABLED and AMP_DTYPE is torch.float16))
    def process_batch(img_r, label_r, img_f, label_f):
        start_time = time.time()
        optim.zero_grad(set_to_none=True)

        # ----- forward in mixed precision -----
        with autocast(enabled=AMP_ENABLED, dtype=AMP_DTYPE):
            loss_1 = ls.loss(model, vae, diffusion, device, img_r, label_f)
            loss_r = ls.loss(model, vae, diffusion, device, img_r, label_r)
            loss_f = ls.loss(model, vae, diffusion, device, img_f, label_f)

        # ----- higher-order grads (build graph) -----
        # These grads are w.r.t. trainable_params; keep graph for the final backward.
        gf = torch.cat([
            g.reshape(-1) for g in torch.autograd.grad(
                outputs=loss_f,
                inputs=trainable_params,
                retain_graph=True,
                create_graph=True
            )
        ])
        gr = torch.cat([
            g.reshape(-1) for g in torch.autograd.grad(
                outputs=loss_r,
                inputs=trainable_params,
                retain_graph=True,
                create_graph=True
            )
        ])

        # Compute orthogonality term in float32 for numerical stability
        gf32 = gf.float()
        gr32 = gr.float()
        denom = (gf32 @ gf32) * (gr32 @ gr32) + 1e-12  # tiny epsilon for safety
        orth  = ( (gf32 @ gr32) ** 2 ) / denom

        # Combine losses (loss_1 may be fp16/bf16; orth is fp32) → result will be fp32
        loss = loss_1 + orthogonality_weight * orth

        # ----- backward with GradScaler -----
        # Scale the final scalar loss (which includes the create_graph path).
        scaled_loss = scaler.scale(loss) if scaler.is_enabled() else loss
        scaled_loss.backward()

        # (Optional) grad clipping after unscale
        if scaler.is_enabled():
            scaler.unscale_(optim)
        # torch.nn.utils.clip_grad_norm_(trainable_params, max_norm)  # if you want it

        # Step
        scaler.step(optim)
        scaler.update()


        elapsed_time = time.time() - start_time
        return float(loss.detach()), elapsed_time, float(orth.detach())

    return process_batch


def get_logger(model, vae, diffusion, identifier, csv_file, log_interval, forget_class, z_random, **gen_kwargs):
    """
    Return a function that logs a single row of results  (loss, time, and class counts) to the CSV file.

    Parameters:
    model (nn.Module): The DiT model.
    vae (nn.Module): The VAE model.
    diffusion (nn.Module): The diffusion model.
    identifier (nn.Module): The image classifier.
    csv_file (str): Path to the CSV file.
    log_interval (int): Interval at which to log.
    forget_class (int): The class to forget.
    z_random (torch.Tensor): The random noise vector.
    **gen_kwargs (dict): Additional keyword arguments to pass to dit.generate_cfg.

    Returns:
    A function that takes in step, losses, elapsed_time, and logs them to the CSV file.
    """
    device = vae.device
    gen_kwargs['class_id'] = forget_class
    gen_kwargs['device'] = device
    gen_kwargs['show'] = False
    gen_kwargs['noise'] = z_random
    gen_kwargs['n_samples'] = z_random.shape[0] // 2
  
    # @ut.timer
    def log_results(step, losses, elapsed_time):
        if step % log_interval == 0:
            gen_imgs = dit.generate_cfg_steady_fast(model, vae, diffusion, **gen_kwargs).clone().detach()
            class_count = cl.identify(identifier, gen_imgs, forget_class, device) / gen_imgs.shape[0]
             # --- free GPU memory ---
            del gen_imgs
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            elif device.type == "mps":
                # no explicit empty_cache API, but you can force a GC
                gc.collect()
            # Write row to the CSV file
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([step] + [losses[0]] + [elapsed_time] + [1 - class_count, class_count] + [losses[1]])
            # return gen_imgs
    return log_results


def train(model_path, folder, num_steps, batch_size, save_steps=None, collect_interval='epoch', log_interval=10,\
          uniformity_weight=0., orthogonality_weight=1000., exchange_classes=[208], forget_class=207,\
          img_ext='jpg', data_path='../../data/ImageNet-1k/2012', imagenet_json_path='../../data/ImageNet-1k/imagenet_1k.json', 
          n_samples=100, device='cuda', diffusion_steps=64, freeze_K=4, unfreeze_last=False, **gen_kwargs):
    """
    UNO on a diffusion-based image synthesis model on the ImageNet dataset.

    Parameters
    ----------
    model_path : str
        Path to a saved model or a model instance.
    folder : str
        Folder to store results.
    num_steps : int
        Number of training steps.
    batch_size : int
        Batch size for training.
    save_steps : int or None, optional
        Interval at which to save the model. Defaults to None, which means to never save.
    collect_interval : str, optional
        Interval at which to collect samples. Must be 'epoch', 'step', or None. Defaults to 'epoch'.
    log_interval : int, optional
        Interval at which to log results. Defaults to 10.
    uniformity_weight : float, optional
        Weight for the uniformity loss. Defaults to 0.
    orthogonality_weight : float, optional        
        Weight for the orthogonality loss. Defaults to 1000.
    exchange_classes : list, optional
        List of class indices to exchange during training. Defaults to [208].
    forget_class : int, optional
        Class index to forget during training. Defaults to 207.
    img_ext : str, optional
        Extension to use for saved images. Defaults to 'jpg'.
    data_path : str, optional
        Path to the ImageNet dataset. Defaults to '../../data/ImageNet-1k/2012'.
    imagenet_json_path : str, optional
        Path to the ImageNet JSON file. Defaults to '../../data/ImageNet-1k/imagenet_1k.json'.
    n_samples : int, optional
        Number of samples to generate at each logging step. Defaults to 100.
    device : str, optional
        Device on which to train the model. Defaults to 'cuda'.
    diffusion_steps : int, optional
        Number of diffusion steps to use. Defaults to 64.
    freeze_K : int, optional
        Number of AdaLN layers to keep during training. Defaults to 4.
    unfreeze_last : bool, optional
        Whether to unfreeze the last layer during training. Defaults to False.
    **gen_kwargs : dict, optional
        Additional keyword arguments to pass to `get_processor`, `get_logger`, `get_saver`, and `get_collector`.

    Returns
    -------
    None
    """
    # ---------------------------------------------------
    # Setup
    # ---------------------------------------------------
    model, vae, diffusion, dataloader, optim, z_random, identifier, sample_dir, checkpoint_dir, epoch_length, epochs,\
           num_steps, save_steps, collect_interval, log_interval, csv_file, device, grid_size, trainable_params \
    = tt.init(model_path, folder, num_steps, batch_size,  save_steps=save_steps, collect_interval=collect_interval,\
           log_interval=log_interval, uniformity_weight=uniformity_weight, orthogonality_weight=orthogonality_weight,\
           exchange_classes=exchange_classes, forget_class=forget_class, img_ext=img_ext,  data_path=data_path, 
           imagenet_json_path=imagenet_json_path,n_samples=n_samples, device=device, diffusion_steps=diffusion_steps,
           freeze_K=freeze_K, unfreeze_last=unfreeze_last)
    # Add a new column to the CSV file
    with open(csv_file, "r+") as f:
        hdr = f.readline().strip() + ",Orthogonality Loss"
        f.seek(0); f.write(hdr + "\n"); f.truncate()
    process_batch = get_processor(model, vae, diffusion, device, optim, trainable_params, orthogonality_weight) 
    if not getattr(process_batch, "_ckpt_patched", False):
        _orig_checkpoint = _cp.checkpoint
        _orig_ckpt_seq   = _cp.checkpoint_sequential

        def _checkpoint_no_reentrant(*args, **kwargs):
            kwargs.setdefault("use_reentrant", False)
            return _orig_checkpoint(*args, **kwargs)

        def _ckpt_seq_no_reentrant(*args, **kwargs):
            kwargs.setdefault("use_reentrant", False)
            return _orig_ckpt_seq(*args, **kwargs)

        _cp.checkpoint = _checkpoint_no_reentrant
        _cp.checkpoint_sequential = _ckpt_seq_no_reentrant
        process_batch._ckpt_patched = True   
    log_results = get_logger(model, vae, diffusion, identifier, csv_file, log_interval, forget_class, z_random, **gen_kwargs)
    save = tt. get_saver(model, save_steps, checkpoint_dir, epoch_length)
    collect_samples = tt.get_collector(sample_dir, collect_interval, grid_size, identifier, img_ext)   


    # ---------------------------------------------------
    # Main training loop
    # ---------------------------------------------------
    global_step, done = 0, False
    for _ in tqdm(range(1, epochs + 1), desc="Epochs"):
        for batch_retain, batch_forget in zip(dataloader['retain'], dataloader['forget']):
            global_step += 1
            img_retain, label_retain = batch_retain[0].to(device), batch_retain[1].to(device)
            img_forget, label_forget = batch_forget[0].to(device), batch_forget[1].to(device)
            # -- Process a single batch
            loss, elapsed_time, orth = process_batch(img_retain, label_retain, img_forget, label_forget)
            generated_img = log_results(step=global_step, losses=[loss, orth], elapsed_time=elapsed_time)
            save(step=global_step)
            collect_samples(generated_img, step=global_step)
            if global_step >= num_steps:
                done = True
                break
        if done:
            break
    viz.summarize_training(folder)





