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



def train(model='./vae.pth', folder='/.', num_steps=100, batch_size=100, latent_dim=512, save_steps=None, collect_interval='epoch', log_interval=10,\
          kl_weight=1., uniformity_weight=1e4, orthogonality_weight=1e5, forget_weight=0., all_classes=[0, 1], forget_class=1,\
          img_ext='jpg', classifier_path="../../data/CelebA/cnn/cnn_10.pth",  data_path="../../data/CelebA/dataset", max_data=None, **viz_kwargs):
    
    """
    Train a Variational Autoencoder (VAE) model using orthogonal-surgery technique.

    Parameters:
    model (str): The path to the model to be loaded. Default is './vae.pth'.
    folder (str): Directory where samples and checkpoints will be saved. Default is '/'.
    num_steps (int): Total number of training steps. Default is 100.
    batch_size (int): Batch size for training. Default is 100.
    latent_dim (int): Dimensionality of the latent space. Default is 512.
    save_steps (list or None): List of steps at which to save the model. Default is None.
    collect_interval (str or int): Interval for collecting samples. Default is 'epoch'.
    log_interval (int): Interval for logging metrics. Default is 10.
    kl_weight (float): Weight for the KL divergence loss term. Default is 1.
    uniformity_weight (float): Weight for the uniformity loss term. Default is 1e4.
    orthogonality_weight (float): Weight for the orthogonality loss term. Default is 1e5.
    forget_weight (float): Weight for the forget loss term. Default is 0.
    all_classes (list): List of classes to be used for training. Default is [0, 1].
    forget_class (int): Class to be forgotten. Default is 1.
    img_ext (str): File extension for saved images. Default is 'jpg'.
    classifier_path (str): Path to the classifier model. Default is "../../data/CelebA/cnn/cnn_10.pth".
    data_path (str): Path to the data folder. Default is "../../data/CelebA/dataset".
    max_data (int or None): Maximum number of data points to use. If None, use all data. Default is None.
    **viz_kwargs: Additional keyword arguments for visualization functions.

    Returns:
    None
    """
    # ---------------------------------------------------
    # Setup
    # ---------------------------------------------------
    net, dataloader, optim, z_random, identifier, sample_dir, checkpoint_dir, epoch_length, epochs,\
    num_steps, save_steps, collect_interval, log_interval, csv_file, device, grid_size \
    = vt.init(model, folder, num_steps, batch_size, latent_dim=latent_dim, save_steps=save_steps, collect_interval=collect_interval,\
              log_interval=log_interval, kl_weight=kl_weight, uniformity_weight=uniformity_weight, orthogonality_weight=orthogonality_weight,\
              forget_weight=forget_weight, all_classes=all_classes, forget_class=forget_class, img_ext=img_ext, classifier_path=classifier_path,\
              train_mode='orthogonal-surgery', data_path=data_path, max_data=max_data)
    process_batch_odd = vo.get_processor(net, ut.get_trainable_params(net), identifier, z_random, (kl_weight, uniformity_weight, orthogonality_weight, forget_weight), optim, all_classes, forget_class)    
    process_batch_even = vs.get_processor(net, ut.get_trainable_params(net), identifier, z_random, (kl_weight, uniformity_weight), optim, all_classes, forget_class)
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
            img_retain = img_retain.to(device)
            img_forget = img_forget.to(device)  
            # -- Process a single batch
            if global_step % 2 == 0:
                rec_loss, kl_loss, unif_loss, orth_loss, generated_img, logits, elapsed_time = process_batch_even(img_retain, img_forget)
            else:
                rec_loss, kl_loss, unif_loss, orth_loss, generated_img, logits, elapsed_time = process_batch_odd(img_retain, img_forget)
            loss = rec_loss + kl_weight * kl_loss + uniformity_weight * unif_loss + orthogonality_weight * orth_loss
            real_img, _ = next(iter(dataloader['original']))
            real_img = real_img.to(device)
            log_results(step=global_step, losses=[rec_loss, kl_loss, unif_loss, orth_loss, loss], elapsed_time=elapsed_time, real_img=real_img, generated_img=generated_img, logits=logits)
            save(step=global_step)
            collect_samples(generated_img, step=global_step)
    viz_kwargs.update({"folder": folder})
    viz.summarize_training(**viz_kwargs) 
