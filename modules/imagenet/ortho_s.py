import train as tt
import ortho as uno
import surgery as ts
import torch, time
import loss as ls
from torch.autograd import grad
from tqdm import tqdm
import viz



def train(model_path, folder, num_steps, batch_size, save_steps=None, collect_interval='epoch', log_interval=10, learning_rate=1e-4,\
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
        Additional keyword arguments to pass to the generator.

    Returns
    -------
    None
    """
    # ---------------------------------------------------
    # Setup
    # ---------------------------------------------------
    model, vae, diffusion, dataloader, optim, z_random, identifier, sample_dir, checkpoint_dir, epoch_length, epochs,\
           num_steps, save_steps, collect_interval, log_interval, csv_file, device, grid_size, trainable_params \
    = tt.init(model_path, folder, num_steps, batch_size,  save_steps=save_steps, collect_interval=collect_interval, learning_rate=learning_rate,\
           log_interval=log_interval, uniformity_weight=uniformity_weight, orthogonality_weight=orthogonality_weight,\
           exchange_classes=exchange_classes, forget_class=forget_class, img_ext=img_ext,  data_path=data_path, 
           imagenet_json_path=imagenet_json_path,n_samples=n_samples, device=device, diffusion_steps=diffusion_steps,
           freeze_K=freeze_K, unfreeze_last=unfreeze_last)
    process_batch_odd = uno.get_processor(model, vae, diffusion, device, optim, trainable_params, orthogonality_weight)
    process_batch_even = ts.get_processor(model, vae, diffusion, device, optim, trainable_params)

    tt.patch_checkpoint_nonreentrant()   

    log_results = tt.get_logger(model, vae, diffusion, identifier, csv_file, log_interval, forget_class, z_random, **gen_kwargs)
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
            if global_step % 2 == 1:
                loss, elapsed_time, _ = process_batch_odd(img_retain, label_retain, img_forget, label_forget)
            else:
                loss, elapsed_time = process_batch_even(img_retain, label_retain, img_forget, label_forget)
            generated_img = log_results(step=global_step, losses=[loss], elapsed_time=elapsed_time)
            save(step=global_step)
            collect_samples(generated_img, step=global_step)
            if global_step >= num_steps:
                done = True
                break
        if done:
            break
    viz.summarize_training(folder)





