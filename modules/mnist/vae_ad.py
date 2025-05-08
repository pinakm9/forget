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


def get_processor_ascent(net, trainable_params, identifier, z_random, weights, optim, all_digits, forget_digit):
    """
    Returns a function that computes gradients of the loss with respect to the parameters of the VAE and performs a gradient ascent step.

    The loss is a weighted sum of the reconstruction loss, KL divergence, and uniformity loss.
    The uniformity loss is computed using the logits of the identifier network.
    The function returns the reconstruction loss, KL divergence, uniformity loss, orthogonality of the gradients, generated image, and the logits of the identifier network.
    The function also returns the time elapsed for computing the gradients.

    Parameters:
        net: the VAE network
        trainable_params: the parameters of the VAE that are to be optimized
        identifier: the identifier network
        z_random: the random noise vector used to generate an image
        weights: the weights of the loss function
        optim: the optimizer used to optimize the parameters of the VAE
        all_digits: all the digits in the dataset
        forget_digit: the digit to be forgotten
    """
    def process_batch(real_img_retain, real_img_forget):
        kl_weight, uniformity_weight = weights
        
        time_0 = time.time()
        optim.zero_grad()

    
        real_img_forget = real_img_forget.view(real_img_forget.shape[0], -1).to(net.device)
        real_img_retain = real_img_retain.view(real_img_retain.shape[0], -1).to(net.device)
        reconstructed_forget, mu_forget, logvar_forget = net(real_img_forget)
        reconstructed_retain, mu_retain, logvar_retain = net(real_img_retain)

        
        rec_forget = vl.reconstruction_loss(reconstructed_forget, real_img_forget)
        rec_retain = vl.reconstruction_loss(reconstructed_retain, real_img_retain)
        kl_forget = vl.kl_div(mu_forget, logvar_forget)
        kl_retain = vl.kl_div(mu_retain, logvar_retain)
        time_1 = time.time()


        generated_img = net.decoder(z_random)
        logits = identifier(generated_img)
        uniformity = vl.uniformity_loss_surgery(logits, all_digits=all_digits, forget_digit=forget_digit)
        time_2 = time.time()

        
        loss_forget = rec_forget + kl_weight * kl_forget + uniformity_weight * uniformity
        gf = torch.cat([g.view(-1) for g in grad(outputs=loss_forget, inputs=trainable_params, retain_graph=True)])
        time_3 = time.time()
        
        loss_retain = rec_retain + kl_weight * kl_retain + uniformity_weight * uniformity
        gr = torch.cat([g.view(-1) for g in grad(outputs=loss_retain, inputs=trainable_params, retain_graph=True)]) 


        # Remove the component of gr that is parallel to g1.
        # (An epsilon is added to avoid division by zero.)
        gfgr = gf @ gr
        gfgf = gf @ gf

        gr = gr - gf * gfgr / gfgf 
        time_4 = time.time()

        # Reassign gradients to the parameters.
        idx = 0
        for p in trainable_params:
            numel = p.numel()
            p.grad = -gf[idx: idx + numel].view(p.shape)
            idx += numel 

        optim.step()
        time_final = time.time() 

        elapsed_time = (time_1 - time_0) + float(uniformity_weight != 0) * (time_2 - time_1) + (time_3 - time_2) + (time_final - time_4)
        orth = gfgr**2 / (gfgf * (gr @ gr))
        return (rec_retain).item(), (kl_retain).item(),  uniformity.item(), orth.item(), generated_img, logits, elapsed_time

    return process_batch




def get_processor_descent(net, trainable_params, identifier, z_random, weights, optim, all_digits, forget_digit):
    def process_batch(real_img_retain, real_img_forget):
        """
        Computes gradients of the loss with respect to the parameters of the VAE and performs a gradient descent step.

        The loss is a weighted sum of the reconstruction loss, KL divergence, and uniformity loss.
        The uniformity loss is computed using the logits of the identifier network.
        The function returns the reconstruction loss, KL divergence, uniformity loss, orthogonality of the gradients, generated image, and the logits of the identifier network.
        The function also returns the time elapsed for computing the gradients.

        Parameters:
            real_img_retain: the real images to be retained
            real_img_forget: the real images to be forgotten

        Returns:
            rec_retain: the reconstruction loss of the retained images
            kl_retain: the KL divergence of the retained images
            uniformity: the uniformity loss
            orth: the orthogonality of the gradients
            generated_img: the generated image
            logits: the logits of the identifier network
            elapsed_time: the time elapsed for computing the gradients
        """
        kl_weight, uniformity_weight = weights
        
        time_0 = time.time()
        optim.zero_grad()

    
        real_img_forget = real_img_forget.view(real_img_forget.shape[0], -1).to(net.device)
        real_img_retain = real_img_retain.view(real_img_retain.shape[0], -1).to(net.device)
        reconstructed_forget, mu_forget, logvar_forget = net(real_img_forget)
        reconstructed_retain, mu_retain, logvar_retain = net(real_img_retain)

        
        rec_forget = vl.reconstruction_loss(reconstructed_forget, real_img_forget)
        rec_retain = vl.reconstruction_loss(reconstructed_retain, real_img_retain)
        kl_forget = vl.kl_div(mu_forget, logvar_forget)
        kl_retain = vl.kl_div(mu_retain, logvar_retain)
        time_1 = time.time()


        generated_img = net.decoder(z_random)
        logits = identifier(generated_img)
        uniformity = vl.uniformity_loss_surgery(logits, all_digits=all_digits, forget_digit=forget_digit)
        time_2 = time.time()

        
        loss_forget = rec_forget + kl_weight * kl_forget + uniformity_weight * uniformity
        gf = torch.cat([g.view(-1) for g in grad(outputs=loss_forget, inputs=trainable_params, retain_graph=True)])
        time_3 = time.time()
        
        loss_retain = rec_retain + kl_weight * kl_retain + uniformity_weight * uniformity
        gr = torch.cat([g.view(-1) for g in grad(outputs=loss_retain, inputs=trainable_params, retain_graph=True)]) 


        # Remove the component of gr that is parallel to g1.
        # (An epsilon is added to avoid division by zero.)
        gfgr = gf @ gr
        gfgf = gf @ gf

        gr = gr - gf * gfgr / gfgf 
        time_4 = time.time()

        # Reassign gradients to the parameters.
        idx = 0
        for p in trainable_params:
            numel = p.numel()
            p.grad = gr[idx: idx + numel].view(p.shape)
            idx += numel 

        optim.step()
        time_final = time.time() 

        elapsed_time = (time_1 - time_0) + float(uniformity_weight != 0) * (time_2 - time_1) + (time_3 - time_2) + (time_final - time_4)
        orth = gfgr**2 / (gfgf * (gr @ gr))
        return (rec_retain).item(), (kl_retain).item(),  uniformity.item(), orth.item(), generated_img, logits, elapsed_time

    return process_batch



def train(model, folder, num_steps, batch_size, latent_dim=2, save_steps=None, collect_interval='epoch', log_interval=10,\
          kl_weight=1., uniformity_weight=1e4, all_digits=list(range(10)), forget_digit=1,\
          img_ext='jpg', classifier_path="../data/MNIST/classifiers/MNISTClassifier.pth", data_path='../../data/MNIST', **viz_kwargs):
    """
    Train the VAE on MNIST digits, with a custom loop to alternate between ascent and descent steps.

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
    all_digits : list, optional
        List of all digits to use. Defaults to list(range(10)).
    forget_digit : int, optional

        Digit to forget. Defaults to 1.
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
           log_interval=log_interval, kl_weight=kl_weight, uniformity_weight=uniformity_weight, orthogonality_weight=0.,\
           all_digits=all_digits, forget_digit=forget_digit, img_ext=img_ext, classifier_path=classifier_path, train_mode='ascent-descent', data_path=data_path)
    process_batch_odd = get_processor_ascent(net, ut.get_trainable_params(net), identifier, z_random, (kl_weight, uniformity_weight), optim, all_digits, forget_digit)    
    process_batch_even = get_processor_descent(net, ut.get_trainable_params(net), identifier, z_random, (kl_weight, uniformity_weight), optim, all_digits, forget_digit)
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
            loss = rec_loss + kl_weight * kl_loss + uniformity_weight * unif_loss 
            real_img, _ = next(iter(dataloader['original']))
            real_img = real_img.view(real_img.shape[0], -1).to(device)
            log_results(step=global_step, losses=[rec_loss, kl_loss, unif_loss, orth_loss, loss], elapsed_time=elapsed_time, real_img=real_img, generated_img=generated_img, logits=logits)
            save(step=global_step)
            collect_samples(generated_img, step=global_step)
    viz_kwargs.update({"folder": folder})
    viz.summarize_training(**viz_kwargs) 