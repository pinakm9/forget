import torch
from tqdm import tqdm
import numpy as np
import os, sys
import time
from torch.autograd import grad


sys.path.append(os.path.abspath('../modules'))
import utility as ut
import vae_loss as vl
import vae_train as vt
import vae_ortho as vo
import vae_viz as viz


def get_processor(net, trainable_params, identifier, z_random, weights, optim, all_classes, forget_class):
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
    weights (tuple): Contains weights for KL divergence and uniformity loss.
    optim (torch.optim.Optimizer): Optimizer for the VAE.
    all_classes (list): List of all class labels.
    forget_class (int): The class label to forget.

    Returns:
    function: A function that takes a batch of images to retain and forget, and returns the reconstruction loss, 
              KL divergence, uniformity loss, orthogonality measure, generated image, logits, and elapsed time.
    """

    # @ut.timer
    def process_batch(real_img_retain, real_img_forget):
        kl_weight, uniformity_weight = weights
        
        time_0 = time.time()
        optim.zero_grad()

    
        reconstructed_forget, mu_forget, logvar_forget = net(real_img_forget)
        reconstructed_retain, mu_retain, logvar_retain = net(real_img_retain)

        
        rec_forget = vl.reconstruction_loss(reconstructed_forget, real_img_forget)
        rec_retain = vl.reconstruction_loss(reconstructed_retain, real_img_retain)
        kl_forget = vl.kl_div(mu_forget, logvar_forget)
        kl_retain = vl.kl_div(mu_retain, logvar_retain)
        time_1 = time.time()

        generated_img = net.decode(z_random)
        logits = identifier(generated_img)
        uniformity = vl.uniformity_loss_surgery(logits, all_classes=all_classes, forget_class=forget_class)
        time_2 = time.time()

        loss_forget = rec_forget + kl_weight * kl_forget + uniformity_weight * uniformity
        loss_retain = rec_retain + kl_weight * kl_retain + uniformity_weight * uniformity

        gf = torch.cat([g.view(-1) for g in grad(outputs=loss_forget, inputs=trainable_params, retain_graph=True)])
        gr = torch.cat([g.view(-1) for g in grad(outputs=loss_retain, inputs=trainable_params, retain_graph=True)]) 


        # Remove the component of gr that is parallel to gf
        gfgr = gf @ gr
        gfgf = gf @ gf

        gr = gr - gf * gfgr / gfgf 

        # Reassign gradients to the parameters.
        idx = 0
        for p in trainable_params:
            numel = p.numel()
            p.grad = gr[idx: idx + numel].view(p.shape)
            idx += numel 

        optim.step()
        time_final = time.time() 

        elapsed_time = (time_1 - time_0) + float(uniformity_weight != 0) * (time_2 - time_1) + (time_final - time_2)
        orth = gfgr**2 / (gfgf * (gr @ gr))
        return (rec_retain).item(), (kl_retain).item(),  uniformity.item(), orth.item(), generated_img, logits, elapsed_time

    return process_batch





def train(model, folder, num_steps, batch_size, latent_dim=512, save_steps=None, collect_interval='epoch', log_interval=10,\
          kl_weight=1., uniformity_weight=0., all_classes=[0, 1], forget_class=1,\
          img_ext='jpg', classifier_path="../../data/CelebA/cnn/cnn_10.pth",  data_path="../../data/CelebA/dataset", max_data=None, **viz_kwargs):
    """
    Train a VAE with an additional classification loss term using the technique
    from the paper "Orthogonalizing Convolutional Neural Networks with Orthogonal
    1x1 Convolutions" [1].

    :param model: The VAE model to train.
    :type model: :py:class:`torch.nn.Module`
    :param folder: The folder to save the results to.
    :type folder: str
    :param num_steps: The number of steps to train for.
    :type num_steps: int
    :param batch_size: The batch size to use.
    :type batch_size: int
    :param latent_dim: The number of latent dimensions to use.
    :type latent_dim: int
    :param save_steps: The number of steps to save the model at.
    :type save_steps: int
    :param collect_interval: The interval at which to collect samples.
    :type collect_interval: str
    :param log_interval: The interval at which to log the loss.
    :type log_interval: int
    :param kl_weight: The weight of the KL loss term.
    :type kl_weight: float
    :param uniformity_weight: The weight of the uniformity loss term.
    :type uniformity_weight: float
    :param all_classes: The list of all classes.
    :type all_classes: list
    :param forget_class: The class to forget.
    :type forget_class: int

    :param img_ext: The image extension to use.
    :type img_ext: str
    :param classifier_path: The path to the classifier.
    :type classifier_path: str
    :param data_path: The path to the dataset.
    :type data_path: str
    :param max_data: The maximum amount of data to use.

    :type max_data: int
    :param **viz_kwargs: The keyword arguments for the visualizer.
    :type viz_kwargs: dict

    References:
    [1] https://arxiv.org/abs/1901.08428
    """
    # ---------------------------------------------------
    # Setup
    # ---------------------------------------------------
    net, dataloader, optim, z_random, identifier, sample_dir, checkpoint_dir, epoch_length, epochs,\
    num_steps, save_steps, collect_interval, log_interval, csv_file, device, grid_size \
    = vt.init(model, folder, num_steps, batch_size, latent_dim=latent_dim, save_steps=save_steps, collect_interval=collect_interval,\
           log_interval=log_interval, kl_weight=kl_weight, uniformity_weight=uniformity_weight, orthogonality_weight=0.,\
           all_classes=all_classes, forget_class=forget_class, img_ext=img_ext, classifier_path=classifier_path, train_mode='orthogonal', data_path=data_path, max_data=max_data)
    process_batch = get_processor(net, ut.get_trainable_params(net), identifier, z_random, (kl_weight, uniformity_weight), optim, all_classes, forget_class)    
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
            img_forget = img_forget.to(device)
            img_retain = img_retain.to(device)
            # -- Process a single batch
            rec_loss, kl_loss, unif_loss, orth_loss, generated_img, logits, elapsed_time = process_batch(img_retain, img_forget)
            loss = rec_loss + kl_weight * kl_loss + uniformity_weight * unif_loss 
            real_img, _ = next(iter(dataloader['original']))
            real_img = real_img.to(device)
            log_results(step=global_step, losses=[rec_loss, kl_loss, unif_loss, orth_loss, loss], elapsed_time=elapsed_time, real_img=real_img, generated_img=generated_img, logits=logits)
            save(step=global_step)
            collect_samples(generated_img, step=global_step)
    viz_kwargs.update({"folder": folder})
    viz.summarize_training(**viz_kwargs) 