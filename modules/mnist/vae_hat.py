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
import vae_viz as viz
import classifier as cl
  


def get_processor(net, trainable_params, identifier, z_random, weights, optim, all_digits, forget_digit):
    """
    Returns a function that processes a batch of images through a VAE network and computes the necessary gradients.

    This function performs a forward and backward pass on a batch of images, calculating the reconstruction loss, 
    KL divergence, and a uniformity loss. It then adjusts the gradients to ensure orthogonality, performs an 
    optimization step, and returns the relevant metrics.

    Parameters:
    net (nn.Module): The VAE model.
    trainable_params (list): List of parameters to optimize.
    identifier (nn.Module): The model used for logits computation.
    z_random (torch.tensor): Random latent codes for the decoder.
    weights (tuple): Contains weights for KL divergence, uniformity loss, and orthogonality loss.
    optim (torch.optim.Optimizer): Optimizer for the VAE.
    all_digits (list): List of all class labels.
    forget_digit (int): The class label to forget.

    Returns:
    function: A function that takes a batch of images to retain and forget, and returns the reconstruction loss, 
              KL divergence, uniformity loss, orthogonality measure, generated image, logits, and elapsed time.
    """
    digits = all_digits
    def process_batch(img_retain, img_forget):
        kl_weight, orthogonality_weight, uniformity_weight, forget_weight = weights
        img_forget = img_forget.view(img_forget.shape[0], -1).to(net.device)
        img_retain = img_retain.view(img_retain.shape[0], -1).to(net.device)
       

        reconstructed_forget, mu_forget, logvar_forget = net(img_forget)
        kl_forget = vl.kl_div(mu_forget, logvar_forget)
        rec_forget = vl.reconstruction_loss(reconstructed_forget, img_forget)
        loss_forget = rec_forget + kl_weight * kl_forget
        
         
        time_0 = time.time()
        generated_img = net.decoder(z_random)
        logits = identifier(generated_img)
        uniformity = vl.uniformity_loss_surgery(logits, all_digits=all_digits, forget_digit=forget_digit) # vl.uniformity_loss(logits, digits)
    
        
        reconstructed_retain, mu_retain, logvar_retain = net(img_retain)
        rec_retain = vl.reconstruction_loss(reconstructed_retain, img_retain)
        kl_retain = vl.kl_div(mu_retain, logvar_retain)
        loss_retain = rec_retain + kl_weight * kl_retain
        time_1 = time.time()

        gf = torch.cat([x.view(-1) for x in grad(outputs=loss_forget + uniformity_weight * uniformity, inputs=trainable_params, retain_graph=True)])
        gr = torch.cat([x.view(-1) for x in grad(outputs=loss_retain + uniformity_weight * uniformity, inputs=trainable_params, retain_graph=True)])
        orth = (gf @ gr)**2 / ((gf @ gf) * (gr @ gr))

        time_2 = time.time()
        loss = forget_weight * loss_forget + loss_retain + uniformity_weight * uniformity
        optim.zero_grad()
        loss.backward()
        optim.step()
        time_final = time.time() 

        elapsed_time = (time_1 - time_0) + (time_final - time_2)

        return (rec_forget + rec_retain).item(), (kl_forget + kl_retain).item(),  uniformity.item(), orth.item(), generated_img, logits, elapsed_time

    return process_batch



def train(model='./vae.pth', folder='/.', num_steps=100, batch_size=100, latent_dim=2, save_steps=None, collect_interval='epoch', log_interval=10,\
          kl_weight=1., uniformity_weight=1e4, orthogonality_weight=0., forget_weight=0., all_digits=list(range(10)), forget_digit=1,\
          img_ext='jpg', classifier_path="../data/MNIST/classifiers/MNISTClassifier.pth", data_path='../../data/MNIST', **viz_kwargs):    
    """
    Train an MNIST VAE with an additional loss term for orthogonal representation.

    Parameters
    ----------
    model : str, optional
        Path to the saved model weights. Default is './vae.pth'.
    folder : str, optional
        Path to the folder where all the outputs will be saved. Default is './'.
    num_steps : int, optional
        Total number of steps to train. Default is 100.
    batch_size : int, optional
        Batch size. Default is 100.
    latent_dim : int, optional
        Dimension of the latent space. Default is 2.
    save_steps : list or None, optional
        List of steps at which to save the model. Set to None to not save at all. Default is [100, 1000, 10000].
    collect_interval : str or int, optional
        Interval at which to collect generated samples. Set to 'epoch' to collect at the end of each epoch.
        Set to a number to collect at every that many steps. Default is 'epoch'.
    log_interval : int, optional

        Interval at which to log the results. Default is 10.
    kl_weight : float, optional
        Weight of the KL loss. Default is 1.
    uniformity_weight : float, optional
        Weight of the uniformity loss. Default is 1e4.
    orthogonality_weight : float, optional
        Weight of the orthogonality loss. Default is 1e5.
    forget_weight : float, optional
        Weight of the forget loss. Default is 0.
    all_digits : list, optional
        List of all digits. Default is [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].
    forget_digit : int, optional
        Digit to forget. Default is 1.
    img_ext : str, optional
        Extension of the images. Default is 'jpg'.
    classifier_path : str, optional
        Path to the saved classifier weights. Default is '../data/MNIST/classifiers/MNISTClassifier.pth'.
    data_path : str, optional
        Path to the MNIST dataset. Default is '../../data/MNIST'.
    **viz_kwargs
        Additional keyword arguments for the visualization function.

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
              train_mode='uniformity', data_path=data_path)
    process_batch = get_processor(net, ut.get_trainable_params(net), identifier, z_random, (kl_weight, uniformity_weight, orthogonality_weight, forget_weight), optim, all_digits, forget_digit)    
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
            rec_loss, kl_loss, unif_loss, orth_loss, generated_img, logits, elapsed_time = process_batch(img_retain, img_forget)
            loss = rec_loss + kl_weight * kl_loss + uniformity_weight * unif_loss + orthogonality_weight * orth_loss
            real_img, _ = next(iter(dataloader['original']))
            real_img = real_img.view(real_img.shape[0], -1).to(device)
            log_results(step=global_step, losses=[rec_loss, kl_loss, unif_loss, orth_loss, loss], elapsed_time=elapsed_time, real_img=real_img, generated_img=generated_img, logits=logits)
            save(step=global_step)
            collect_samples(generated_img, step=global_step)
    viz_kwargs.update({"folder": folder})
    viz.summarize_training(**viz_kwargs) 
