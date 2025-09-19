import json, time, csv, os
import dit
import torch
import loss as ls
import classifier as cl
import save as sv
import matplotlib.pyplot as plt
import numpy as np
import datapipe
from tqdm import tqdm
import viz
import torch.utils.checkpoint as _cp
import gc

def write_config(model, folder, epochs, epoch_length, batch_size,  collect_interval='epoch', log_interval='epoch', orthogonality_weight=None,\
                uniformity_weight=None, forget_weight=None, learning_rate=None, exchange_classes=None, forget_class=None, img_ext='JPEG'):
    sample_dir = f'{folder}/samples'    
    checkpoint_dir = f'{folder}/checkpoints'
    # At the end of the train() function, after training is complete:
    config_data = {
        "training": {
            "epochs": {
                "value": epochs,
                "description": "The number of epochs to train the model."
            },
            "epoch_length": {
                "value": epoch_length,
                "description": "The number of descent steps in an epoch."
            },
            "batch_size": {
                "value": batch_size,
                "description": "Number of samples per training batch."
            },
            "learning_rate": {
                "value": learning_rate,  # This example uses the default for AdamW.
                "description": "Learning rate used by the optimizer."
            },
            "optimizer": {
                "value": "Adam",
                "description": "The type of optimizer used during training."
            },
            "device": { 
                "value": str(next(model.parameters()).device),
                "description": "The device used for training."
            }
        },
        "paths": {
            "sample_dir": {
                "value": sample_dir,
                "description": "Directory where generated sample images are saved."
            },
            "checkpoint_dir": {
                "value": checkpoint_dir,
                "description": "Directory where model checkpoints are stored."
            }
        },
        "experiment": {
            "img_ext": {
                "value": img_ext,
                "description": "File extension for sample images."
            },
            "collect_interval": {
                "value": collect_interval,
                "description": "Frequency of collecting samples."
            },
            "log_interval": {
                "value": log_interval,
                "description": "Frequency of logging training loss."
            }
            
        }
    }

    if orthogonality_weight is not None:
        config_data["training"]["orthogonality_weight"] = {
            "value": orthogonality_weight,
            "description": "Weight of the orthogonality loss."
        }

    if forget_weight is not None:
        config_data["training"]["forget_weight"] = {
            "value": forget_weight,
            "description": "Weight of the forget loss."
        }

    if uniformity_weight is not None:
        config_data["training"]["uniformity_weight"] = {
            "value": uniformity_weight,
            "description": "Weight of the uniformity loss."
        }

    if forget_class is not None:
        config_data["experiment"]["forget_class"] = {
            "value": forget_class,
            "description": "class to forget."
        }

    if exchange_classes is not None:
        config_data["experiment"]["exchange_classes"] = {
            "value": exchange_classes,
            "description": "classes to exchange with."
        }
    
    # Save the configuration to a JSON file in the main folder.
    config_path = f"{folder}/config.json"
    with open(config_path, "w") as config_file:
        json.dump(config_data, config_file, indent=4)




def init_model(path, device):
    """
    Initializes a DiT model from a checkpoint path.

    Parameters
    ----------
    path : str
        The path to the checkpoint file.
    device : torch.device
        The device on which to load the model.

    Returns
    -------
    tuple of (DiT_models, AutoencoderKL)
        The loaded DiT-XL/2 model and the loaded AutoencoderKL model.
    """
    return dit.load_DiT(path, device)



def get_processor(model, vae, diffusion, device, optim):
    """
    Returns a processor function, which performs a forward + backward pass on a single batch.

    The returned processor function takes a single batch of real images as input, and returns the individual loss terms.

    Parameters
    ----------
    model : DiT_XL/2
        The DiT-XL/2 model.
    vae : AutoencoderKL
        The AutoencoderKL model.
    diffusion : DiT_Diffusion
        The DiT_Diffusion model.
    device : torch.device
        The device on which to train the model.
    optim : torch.optim.Optimizer
        The optimizer used during training.

    Returns
    -------
    callable
        The processor function.
    """
    device = vae.device
    amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # @ut.timer
    def process_batch(real_img, label):
        # Forward pass
        start_time =  time.time()
        # net.eval()
        
        optim.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            loss = ls.loss(model, vae, diffusion, device, real_img, label)

        scaler.scale(loss).backward()
        # (optimional) grad clip:
        # scaler.unscale_(optim); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(optim)
        scaler.update()

        elapsed_time = time.time() - start_time

        return loss.item(),  elapsed_time
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
                writer.writerow([step] + losses + [elapsed_time] + [1 - class_count, class_count])
            # return gen_imgs
    return log_results




def get_saver(model, save_steps, checkpoint_dir, epoch_length):
    def save(step):
        """
        Saves the DiT model to a file at the given step.

        Saves the model to a file named DiT_epoch_<epoch number>.pth if step is a multiple of epoch_length,
        otherwise saves to a file named DiT_step_<step number>.pth.

        Parameters:
        step (int): The current step number.
        """
        if step in save_steps:
            if step % epoch_length == 0:
                sv.save_trainable_checkpoint(model, f"{checkpoint_dir}/DiT_epoch_{int(step/epoch_length)}.pth")
            else:
                sv.save_trainable_checkpoint(model, f"{checkpoint_dir}/DiT_step_{step}.pth")
    return save







def get_collector(sample_dir, collect_interval, grid_size, identifier, img_ext='jpg'):
    """
    Collect and save generated ImageNet-style images in a grid,
    with identifier-predicted labels shown on top of each image.
    """
    def collect_samples(generated_img, step):
        if step % collect_interval != 0 or generated_img is None:
            return

        imgs = generated_img.detach().clamp(-1, 1)
        imgs = (imgs + 1) / 2  # [-1,1] -> [0,1]

        # Run identifier
        logits = identifier(imgs).logits               # (B, num_classes)
        labels = logits.argmax(dim=-1).tolist()  # top-1 predictions
        labels_str = [str(i) for i in labels]

        # Convert to numpy (B,H,W,C)
        imgs_np = imgs.permute(0, 2, 3, 1).cpu().numpy()

        # --- Build matplotlib grid ---
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size*2, grid_size*2))
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            ax.axis("off")
            if i < len(imgs_np):
                ax.imshow(imgs_np[i])
                ax.set_title(labels_str[i], fontsize=8)

        filename = os.path.join(sample_dir, f"sample_{step}.{img_ext}")
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close(fig)
        print(f"[step {step}] Saved labeled grid -> {filename}")

    return collect_samples








def init(model_path, folder='.', num_steps=100, batch_size=100, save_steps=None, collect_interval='epoch', log_interval=10, learning_rate=1e-4,\
        uniformity_weight=None, orthogonality_weight=None, forget_weight=None, exchange_classes=None, forget_class=None, img_ext='jpg',\
        train_mode='original', data_path='../../data/ImageNet-1k/2012', imagenet_json_path='../../data/ImageNet-1k/imagenet_class_index.json',\
        n_samples=100, device='cuda', diffusion_steps=64, freeze_K=4, unfreeze_last=False, unfreeze_x_embedder=False, keep_all=False):
    
    model, vae = init_model(model_path, device)
    model.train()
    model, _ = dit.freeze_except_y_and_lastK_adaln(model, freeze_K, unfreeze_last, unfreeze_x_embedder, keep_all)
    diffusion = dit.load_diffusion(diffusion_steps)
    identifier = cl.get_classifier(device)
    sample_dir = f'{folder}/samples'
    checkpoint_dir = f'{folder}/checkpoints'
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    dataloader = {'original': datapipe.get_dataloader_multi(root=data_path, class_ids=exchange_classes, imagenet_json_path=imagenet_json_path, batch_size=batch_size)}
    dataloader['retain'] = dataloader['original']
    dataloader['forget'] = datapipe.get_dataloader(root=data_path, class_id=forget_class, imagenet_json_path=imagenet_json_path, batch_size=batch_size)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.)
    latent_size = 256 // 8
    z_random = torch.randn(2*n_samples, 4, latent_size, latent_size, device=device)
    epoch_length = min(len(dataloader['forget']), len(dataloader['retain'])) #len(dataloader['original']) if train_mode == 'original' else 
    epochs = int(np.ceil(num_steps/epoch_length))
    # num_steps = epochs * epoch_length
    grid_size = 5#int(np.ceil(np.sqrt(batch_size)))

    if isinstance(save_steps, list):
        save_steps = save_steps + [epoch_length, num_steps]
        save_steps = list(set(save_steps))
        save_steps.sort()
    elif save_steps == "epoch":
        save_steps = list(range(epoch_length, num_steps + 1, epoch_length))
    else:
        save_steps = [epoch_length, num_steps]

    if collect_interval == 'epoch':
        collect_interval = epoch_length

    if log_interval == 'epoch':         
        log_interval = epoch_length

    # ---------------------------------------------------
    # Prepare CSV logging
    # ---------------------------------------------------     
    csv_file = f"{checkpoint_dir}/training_log.csv"
    # Extra columns for class distribution + ambiguity
    # header = ["Step", "Reconstruction Loss", "KL Loss", "Uniformity Loss", "Orthogonality Loss", "Total Loss", "Time"]
    # header += ["FID", "IS", "Entropy", "Margin"] + [f"{i} Fraction" for i in range(2)]

    header = ["Step", "Total Loss", "Time"]
    header += [f"{i} Fraction" for i in range(2)]
    # Write CSV header
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
    # Save config
    write_config(model=model, folder=folder, epochs=epochs, epoch_length=epoch_length, batch_size=batch_size,\
                collect_interval=collect_interval, log_interval=log_interval, uniformity_weight=uniformity_weight,\
                orthogonality_weight=orthogonality_weight, forget_weight=forget_weight, learning_rate=learning_rate,\
                exchange_classes=exchange_classes, forget_class=forget_class, img_ext=img_ext)

    return model, vae, diffusion, dataloader, optim, z_random, identifier, sample_dir, checkpoint_dir, epoch_length, epochs,\
           num_steps,save_steps, collect_interval, log_interval, csv_file, device, grid_size, trainable_params 



    
def patch_checkpoint_nonreentrant():
    if getattr(_cp, "_nonreentrant_patched", False):
        return

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
    _cp._nonreentrant_patched = True




def train(model_path, folder, num_steps, batch_size, save_steps=None, collect_interval='epoch', log_interval=10, learning_rate=1e-4,\
          uniformity_weight=0., exchange_classes=[208], forget_class=207, img_ext='jpg', data_path='../../data/ImageNet-1k/2012',\
          imagenet_json_path='../../data/ImageNet-1k/imagenet_1k.json', n_samples=100, device='cuda', diffusion_steps=64,\
          freeze_K=4, unfreeze_last=False, unfreeze_x_embedder=False, keep_all=False, **gen_kwargs):
    """
    Train a diffusion-based image synthesis model on the ImageNet dataset.

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
           num_steps, save_steps, collect_interval, log_interval, csv_file, device, grid_size, trainable_params\
    = init(model_path, folder, num_steps, batch_size,  save_steps=save_steps, collect_interval=collect_interval,\
           log_interval=log_interval, uniformity_weight=uniformity_weight, orthogonality_weight=0., learning_rate=learning_rate,\
           exchange_classes=exchange_classes, forget_class=forget_class, img_ext=img_ext,  data_path=data_path, 
           imagenet_json_path=imagenet_json_path,n_samples=n_samples, device=device, diffusion_steps=diffusion_steps,
           freeze_K=freeze_K, unfreeze_last=unfreeze_last, unfreeze_x_embedder=unfreeze_x_embedder, keep_all=keep_all)
    process_batch = get_processor(model, vae, diffusion, device, optim)    
    log_results = get_logger(model, vae, diffusion, identifier, csv_file, log_interval, forget_class, z_random, **gen_kwargs)
    save = get_saver(model, save_steps, checkpoint_dir, epoch_length)
    collect_samples = get_collector(sample_dir, collect_interval, grid_size, identifier, img_ext)   

    # ---------------------------------------------------
    # Main training loop
    # ---------------------------------------------------
    global_step, done = 0, False
    for _ in tqdm(range(1, epochs + 1), desc="Epochs"):
        for real_img, label in dataloader['original']:
            global_step += 1
            real_img = real_img.to(device)
            # -- Process a single batch
            loss, elapsed_time = process_batch(real_img, label)
            generated_img = log_results(step=global_step, losses=[loss], elapsed_time=elapsed_time)
            # save(step=global_step)
            collect_samples(generated_img, step=global_step)
            if global_step >= num_steps:
                done = True
                break
        if done:
            break
    sv.save_trainable_checkpoint(model, f"{checkpoint_dir}/DiT_step_{global_step}.pth")
    viz.summarize_training(folder)
