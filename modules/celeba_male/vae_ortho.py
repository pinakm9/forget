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
import vae_viz as viz
import classifier as cl  



def get_processor(net, trainable_params, identifier, z_random, weights, optim, all_classes, forget_class):
    """
    Returns a function that performs a forward + backward pass on a single batch of images from the generator and two classes.
    
    Parameters:
    net (nn.Module): the VAE model
    trainable_params (list): list of parameters to be optimized
    identifier (nn.Module): the identifier model
    z_random (torch.tensor): a tensor of random latent codes
    weights (list): a list of length 4 containing the weights of the reconstruction loss, KL divergence, uniformity loss, and forgetting loss
    optim (torch.optim.Optimizer): the optimizer for the VAE
    all_classes (list): a list of all class labels
    forget_class (int): the class label to forget
    
    Returns:
    process_batch (function): a function that takes a batch of images from the generator, retain and forget classes, and returns the reconstruction loss, KL divergence, uniformity loss, generated image, logits of the identifier, and elapsed time.
    """
    classes = all_classes
    # @ut.timer
    def process_batch(img_retain, img_forget):
        kl_weight, orthogonality_weight, uniformity_weight, forget_weight = weights
        
        time_0 = time.time()

        reconstructed_forget, mu_forget, logvar_forget = net(img_forget)
        reconstructed_retain, mu_retain, logvar_retain = net(img_retain)
        time_1 = time.time()

        generated_img = net.decode(z_random)
        logits = identifier(generated_img)
        uniformity = vl.uniformity_loss_surgery(logits, all_classes=all_classes, forget_class=forget_class) # vl.uniformity_loss(logits, classes)
        time_2 = time.time()
    
        rec_forget = vl.reconstruction_loss(reconstructed_forget, img_forget)
        rec_retain = vl.reconstruction_loss(reconstructed_retain, img_retain)
        kl_forget = vl.kl_div(mu_forget, logvar_forget)
        kl_retain = vl.kl_div(mu_retain, logvar_retain)

        loss_forget = rec_forget + kl_weight * kl_forget
        loss_retain = rec_retain + kl_weight * kl_retain

        gf = torch.cat([x.view(-1) for x in grad(outputs=loss_forget + uniformity_weight * uniformity, inputs=trainable_params, retain_graph=True)])
        gr = torch.cat([x.view(-1) for x in grad(outputs=loss_retain + uniformity_weight * uniformity, inputs=trainable_params, retain_graph=True)])
        orth = (gf @ gr)**2 / ((gf @ gf) * (gr @ gr))

        loss = forget_weight * loss_forget + loss_retain + orthogonality_weight * orth + uniformity_weight * uniformity
        optim.zero_grad()
        loss.backward()
        optim.step()
        time_final = time.time() 

        elapsed_time = (time_1 - time_0) + float(uniformity_weight != 0.) * (time_2 - time_1) + (time_final - time_2)

        return (rec_forget + rec_retain).item(), (kl_forget + kl_retain).item(),  uniformity.item(), orth.item(), generated_img, logits, elapsed_time

    return process_batch


def get_logger(identifier, csv_file, log_interval):
    """
    Returns a function that logs training results to a CSV file at specified intervals.

    Parameters:
    identifier: An object used to compute image quality metrics.
    csv_file: The path to the CSV file where results are logged.
    log_interval: The interval at which to log the results.

    Returns:
    A function that logs a row of results, including reconstruction loss, KL divergence,
    uniformity, total loss, elapsed time, image quality metrics, and class statistics, to the CSV file.
    """

    def log_results(step, losses, elapsed_time, real_img, generated_img, logits):
        """
        Log a single row of results (rec, KL, uniformity, total loss, time, and class stats) to the CSV file.
        """
        if step % log_interval == 0:
            img_quality = [] # [cl.frechet_inception_distance(real_img, generated_img, identifier), cl.inception_score(logits).item()]
            # Compute class counts and ambiguities
            class_counts = cl.count_from_logits(logits) / logits.shape[0]
            entropy, margin = cl.ambiguity(logits)
            img_quality += [entropy.mean().item(), margin.mean().item()]

            # Write row to the CSV file
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([step] + losses + [elapsed_time] + img_quality + [1 - class_counts, class_counts])
    return log_results




def train(model='./vae.pth', folder='/.', num_steps=100, batch_size=100, latent_dim=512, save_steps=None, collect_interval='epoch', log_interval=10,\
          kl_weight=1., uniformity_weight=0., orthogonality_weight=1e1, forget_weight=0., all_classes=[0, 1], forget_class=1,\
          img_ext='jpg', classifier_path="../../data/CelebA/cnn/cnn_10.pth",  data_path="../../data/CelebA/dataset", max_data=None, **viz_kwargs):
    """
    Train a VAE model with an orthogonality loss term.

    Parameters:
    model (str): The path to the model to be loaded. Default is './vae.pth'.
    folder (str): The root directory where the samples and checkpoints will be saved. Default is '.'.
    num_steps (int): The total number of training steps. Default is 100.
    batch_size (int): The batch size for training. Default is 100.
    latent_dim (int): The dimensionality of the latent space. Default is 2.
    save_steps (list): A list of steps at which to save the model. Default is [100].
    collect_interval (int or str): The interval at which to collect samples. If 'epoch', this is set to the epoch length. Default is 'epoch'.
    log_interval (int or str): The interval at which to log metrics. If 'epoch', this is set to the epoch length. Default is 10.
    kl_weight (float): The weight for the KL loss term. Default is 1.
    uniformity_weight (float): The weight for the uniformity loss term. Default is 0.
    orthogonality_weight (float): The weight for the orthogonality loss term. Default is 1e5.
    forget_weight (float or None): The weight for the forget loss term. If None, the forget loss is not used. Default is 0.
    all_classes (list or None): The list of classes to be used for training. If None, all classes are used. Default is [0, 1].
    forget_class (int or None): The class to be forgotten. If None, no class is forgotten. Default is 1.
    img_ext (str): The file extension for the saved images. Default is 'jpg'.
    classifier_path (str): The path to the classifier model. Default is "../../data/CelebA/cnn/cnn_10.pth".
    data_path (str): The path to the data folder. Defaults to'../../data/CelebA/dataset'
    max_data (int or None): The maximum number of data points to use. If None, all data points are used. Default is None.

    **viz_kwargs: Additional keyword arguments that can be passed to the visualization functions.
    """
    # ---------------------------------------------------
    # Setup
    # ---------------------------------------------------
    net, dataloader, optim, z_random, identifier, sample_dir, checkpoint_dir, epoch_length, epochs,\
    num_steps, save_steps, collect_interval, log_interval, csv_file, device, grid_size \
    = vt.init(model, folder, num_steps, batch_size, latent_dim=latent_dim, save_steps=save_steps, collect_interval=collect_interval,\
              log_interval=log_interval, kl_weight=kl_weight, uniformity_weight=uniformity_weight, orthogonality_weight=orthogonality_weight,\
              forget_weight=forget_weight, all_classes=all_classes, forget_class=forget_class, img_ext=img_ext, classifier_path=classifier_path,\
              train_mode='orthogonal', data_path=data_path, max_data=max_data)
    process_batch = get_processor(net, ut.get_trainable_params(net), identifier, z_random, (kl_weight, uniformity_weight, orthogonality_weight, forget_weight), optim, all_classes, forget_class)    
    log_results = get_logger(identifier, csv_file, log_interval)
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
            loss = rec_loss + kl_weight * kl_loss + uniformity_weight * unif_loss + orthogonality_weight * orth_loss
            real_img, _ = next(iter(dataloader['original']))
            real_img = real_img.to(device)
            log_results(step=global_step, losses=[rec_loss, kl_loss, unif_loss, orth_loss, loss], elapsed_time=elapsed_time, real_img=real_img, generated_img=generated_img, logits=logits)
            save(step=global_step)
            collect_samples(generated_img, step=global_step)
    viz_kwargs.update({"folder": folder})
    viz.summarize_training(**viz_kwargs) 