import torch
from tqdm import tqdm
import numpy as np
import os, sys, csv
import time
from torch.autograd import grad


sys.path.append(os.path.abspath('../modules'))
import utility as ut
import vae_loss as vl
import vae_train as vt
import vae_ortho as vo
import vae_surgery as vs
import vae_viz as viz
import classifier as cl



def train(model='./vae.pth', folder='/.', num_steps=100, batch_size=100, latent_dim=2, save_steps=None, collect_interval='epoch', log_interval=10,\
          kl_weight=1., uniformity_weight=1e4, orthogonality_weight=1e5, forget_weight=0., all_digits=list(range(10)), forget_digit=1,\
          img_ext='jpg', classifier_path="../data/MNIST/classifiers/MNISTClassifier.pth", data_path='../../data/MNIST', **viz_kwargs):
    
    """
    Train the VAE on MNIST classes, with a custom loop to alternate between ascent and descent steps.
    Also applies surgery to the weights of the VAE after each update step.

    Parameters
    ----------
    model : str or torch.nn.Module
        Path to a saved model or a model itself.
    folder : str
        Folder to store results.
    num_steps : int
        Number of training steps.
    batch_size : int
        Batch size for training.
    latent_dim : int, optional
        Dimensionality of the latent space. Defaults to 2.
    save_steps : int or None, optional
        Interval at which to save the model. Defaults to None, which means to never save.
    collect_interval : str, optional
        Interval at which to collect samples. Must be 'epoch', 'step', or None. Defaults to 'epoch'.
    log_interval : int, optional
        Interval at which to log results. Defaults to 10.
    kl_weight : float, optional
        Weight for the KL loss. Defaults to 1.
    uniformity_weight : float, optional
        Weight for the uniformity loss. Defaults to 1e4.
    orthogonality_weight : float, optional
        Weight for the orthogonality loss. Defaults to 1e5.
    forget_weight : float, optional
        Weight for the forgetting loss. Defaults to 0.
    all_digits : list, optional
        List of all classes to use. Defaults to list(range(10)).
    forget_digit : int, optional

        class to forget. Defaults to 1.

    img_ext : str, optional
        Extension to use for saved images. Defaults to 'jpg'.
    classifier_path : str, optional
        Path to a saved classifier. Defaults to "../data/MNIST/classifiers/MNISTClassifier.pth".
    **viz_kwargs : dict, optional
        Additional keyword arguments to pass to `viz.summarize_training`.

    Returns
    -------
    None
    """
    # ---------------------------------------------------
    # Setup
    # ---------------------------------------------------
    net, dataloader, optim, z_random, identifier, sample_dir, checkpoint_dir, epoch_length, epochs,\
    num_steps, save_steps, collect_interval, log_interval, csv_file, device, grid_size \
    = vt.init(model, folder, num_steps, batch_size, latent_dim=latent_dim, save_steps=save_steps, collect_interval=collect_interval,\
              log_interval=log_interval, kl_weight=kl_weight, uniformity_weight=uniformity_weight, orthogonality_weight=orthogonality_weight,\
              forget_weight=forget_weight, all_digits=all_digits, forget_digit=forget_digit, img_ext=img_ext, classifier_path=classifier_path,\
              train_mode='orthogonal-surgery', data_path=data_path)
    process_batch_odd = vo.get_processor(net, ut.get_trainable_params(net), identifier, z_random, (kl_weight, uniformity_weight, orthogonality_weight, forget_weight), optim, all_digits, forget_digit)    
    process_batch_even = vs.get_processor(net, ut.get_trainable_params(net), identifier, z_random, (kl_weight, uniformity_weight), optim, all_digits, forget_digit)
    log_results = vo.get_logger(identifier, csv_file, log_interval)
    save = vt.get_saver(net, save_steps, checkpoint_dir, epoch_length)
    collect_samples = vt.get_collector(sample_dir, collect_interval, grid_size, img_ext)   
    # ---------------------------------------------------
    # Main training loop
    # ---------------------------------------------------
    global_step = 0
    for _ in tqdm(range(1, epochs + 1), desc="Epochs"):
        for (img_retain, _), (img_forget, _) in zip(dataloader['retain'], dataloader['forget']):
            global_step += 1
            # -- Process a single batch
            if global_step % 2 == 0:
                rec_loss, kl_loss, unif_loss, orth_loss, generated_img, logits, elapsed_time = process_batch_even(img_retain, img_forget)
            else:
                rec_loss, kl_loss, unif_loss, orth_loss, generated_img, logits, elapsed_time = process_batch_odd(img_retain, img_forget)
            loss = rec_loss + kl_weight * kl_loss + uniformity_weight * unif_loss + orthogonality_weight * orth_loss
            real_img, _ = next(iter(dataloader['original']))
            real_img = real_img.view(real_img.shape[0], -1).to(device)
            log_results(step=global_step, losses=[rec_loss, kl_loss, unif_loss, orth_loss, loss], elapsed_time=elapsed_time, real_img=real_img, generated_img=generated_img, logits=logits)
            save(step=global_step)
            collect_samples(generated_img, step=global_step)
    viz_kwargs.update({"folder": folder})
    viz.summarize_training(**viz_kwargs) 
































