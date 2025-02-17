import torch
import torch.nn as nn   
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os, sys, csv, gc, glob
import time
from torch.autograd import grad
from torch.nn.utils import vector_to_parameters
import pandas as pd
import json

sys.path.append(os.path.abspath('../modules'))
import utility as ut
import datapipe
import vae_loss as vl
from vae import VAE
import classifier as cl
import vae_train as vt


def train(model, folder, epochs, batch_size, latent_dim, device, orthogonality_weight=10., log_interval="epoch", kl_weight=1., one_weight=0.):
    """
    Train a VAE model with an additional orthogonality loss term.

    Parameters
    ----------
    model : str or VAE
        The path to the pre-trained VAE model to load, or the pre-trained VAE model itself.
    folder : str
        The root directory to save samples and checkpoints.
    epochs : int
        The number of epochs to train the model.
    batch_size : int
        The batch size.
    latent_dim : int
        The dimension of the latent space.
    device : torch.device
        The device to use for training (e.g. GPU or CPU).
    orthogonality_weight : float, optional
        The factor to multiply the orthogonality loss term by.
    log_interval : str or int, optional
        If 'epoch' (default), logs data every epoch.
        If an integer > 0, logs data every that many gradient descent steps.
    
    Notes
    -----
    This function is similar to the `train` function, but it adds an additional orthogonality
    loss term to the loss function. The orthogonality loss term is the mean of the squared dot
    product of the gradients of the loss with respect to the parameters of the VAE model, for
    each batch of samples. This term encourages the VAE model to learn a linear subspace in the
    latent space, which is useful for disentangling the latent factors.
    """
    categorizer = cl.get_classifier(device=device)
    sample_dir = f'{folder}/samples'
    checkpoint_dir = f'{folder}/checkpoints'
    ut.makedirs(sample_dir, checkpoint_dir)

    # Determine logging mode and write CSV header accordingly.
    if log_interval == 'epoch':
        header = ["Epoch", "Reconstruction Loss", "KL Loss", "Orthogonality Loss", "Total Loss", "Time"]
        csv_file = f"{checkpoint_dir}/training_log_epoch.csv"
    elif isinstance(log_interval, int) and log_interval > 0:
        header = ["Step", "Reconstruction Loss", "KL Loss", "Orthogonality Loss", "Total Loss", "Time"]
        csv_file = f"{checkpoint_dir}/training_log_step.csv"
    else:
        raise ValueError("log_interval must be 'epoch' or a positive integer")
    header += ["Entropy", "Margin", "Ambiguity"] + ["{} Fraction".format(i) for i in range(10)]

    # Write header to CSV file (overwriting any existing file).
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

    z_random = torch.randn((batch_size, latent_dim)).to(device)

    dataloader_one, dataloader_rest = datapipe.MNIST().get_dataloader_one_vs_rest(batch_size)
    if isinstance(model, str):
        net = VAE(latent_dim=latent_dim, device=device).to(device)
        net.load_state_dict(torch.load(model))
    else:
        net = model

    net.train()
    optim = torch.optim.Adam(net.parameters())

    trainable_params = ut.get_trainable_params(net)
    # print(len(trainable_params))


    vt.write_config(model=net, folder=folder, epochs=epochs, batch_size=batch_size, latent_dim=latent_dim,\
                log_interval=log_interval, kl_weight=kl_weight, orthogonality_weight=orthogonality_weight, one_weight=one_weight)

    if log_interval == "epoch":
        # Epoch-level logging.
        for epoch_i in tqdm(range(1, epochs + 1), desc="Epochs"):
            start_time = time.time()
            epoch_rec_loss = 0.0
            epoch_kl_loss = 0.0
            epoch_orth_loss = 0.0
            num_batches = 0

            for (real_img_one, _), (real_img_rest, _) in zip(dataloader_one, dataloader_rest):
                real_img_one = real_img_one.view(real_img_one.shape[0], -1).to(device)
                real_img_rest = real_img_rest.view(real_img_rest.shape[0], -1).to(device)

                reconstructed_one, mu_one, logvar_one = net(real_img_one)
                reconstructed_rest, mu_rest, logvar_rest = net(real_img_rest)

                reconstruction_loss_one = vl.reconstruction_loss(reconstructed_one, real_img_one)
                reconstruction_loss_rest = vl.reconstruction_loss(reconstructed_rest, real_img_rest)
             
                kl_loss_one = vl.kl_div(mu_one, logvar_one)
                kl_loss_rest = vl.kl_div(mu_rest, logvar_rest)
              
                loss_one = reconstruction_loss_one +  kl_weight*kl_loss_one 
                loss_rest = reconstruction_loss_rest + kl_weight*kl_loss_rest

                
                gradients_one = torch.cat([g.view(-1) for g in grad(outputs=loss_one, inputs=trainable_params, create_graph=True, retain_graph=True)])
                gradients_rest = torch.cat([g.view(-1) for g in grad(outputs=loss_rest, inputs=trainable_params, create_graph=True, retain_graph=True)])
                
                orthogonality_loss = (gradients_one * gradients_rest).mean()**2 

                loss = one_weight * loss_one + loss_rest + orthogonality_weight * orthogonality_loss

                optim.zero_grad()
                loss.backward()
                optim.step()

                epoch_rec_loss += (one_weight * reconstruction_loss_one + reconstruction_loss_rest).item()
                epoch_kl_loss += (one_weight * kl_loss_one + kl_loss_rest).item()
                epoch_orth_loss += orthogonality_loss.item()
                num_batches += 1

            with torch.no_grad():
                logits = categorizer(net.decoder(z_random))

            # Compute average losses and time.
            epoch_rec_loss /= num_batches
            epoch_kl_loss /= num_batches
            epoch_orth_loss /= num_batches
            total_loss = epoch_rec_loss + kl_weight * epoch_kl_loss + orthogonality_weight * epoch_orth_loss
            epoch_time = time.time() - start_time

            class_counts = cl.count_from_logits(logits) / batch_size
            entropy, margin, ambiguity = cl.ambiguity(logits)
            entropy, margin, ambiguity = entropy.mean(), margin.mean(), ambiguity.mean()

            # Append epoch results to CSV.
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch_i, epoch_rec_loss, epoch_kl_loss, epoch_orth_loss, total_loss, epoch_time] +\
                                        [entropy.item(), margin.item(), ambiguity.item()] + class_counts.tolist())

            # Save model checkpoint.
            torch.save(net.state_dict(), f"{checkpoint_dir}/vae_{epoch_i}.pth")
    else:
        # Gradient-step interval logging.
        global_step = 0
        interval_rec_loss = 0.0
        interval_kl_loss = 0.0
        interval_orth_loss = 0.0
        count_steps = 0
        interval_start_time = time.time()

        for epoch_i in tqdm(range(1, epochs + 1), desc="Epochs"):
            for (real_img_one, _), (real_img_rest, _) in zip(dataloader_one, dataloader_rest):
                global_step += 1
                count_steps += 1

                real_img_one = real_img_one.view(real_img_one.shape[0], -1).to(device)
                real_img_rest = real_img_rest.view(real_img_rest.shape[0], -1).to(device)

                reconstructed_one, mu_one, logvar_one = net(real_img_one)
                reconstructed_rest, mu_rest, logvar_rest = net(real_img_rest)

                reconstruction_loss_one = vl.reconstruction_loss(reconstructed_one, real_img_one)
                reconstruction_loss_rest = vl.reconstruction_loss(reconstructed_rest, real_img_rest)
                

                kl_loss_one = vl.kl_div(mu_one, logvar_one)
                kl_loss_rest = vl.kl_div(mu_rest, logvar_rest)
             
                loss_one = reconstruction_loss_one +  kl_weight*kl_loss_one 
                loss_rest = reconstruction_loss_rest + kl_weight*kl_loss_rest

                gradients_one = torch.cat([g.view(-1) for g in grad(outputs=loss_one, inputs=trainable_params, create_graph=True, retain_graph=True)])
                gradients_rest = torch.cat([g.view(-1) for g in grad(outputs=loss_rest, inputs=trainable_params, create_graph=True, retain_graph=True)])
              
                orthogonality_loss = (gradients_one * gradients_rest).mean()**2 

                # print(gradients_one.shape, gradients_rest.shape)

                loss = one_weight * loss_one + loss_rest + orthogonality_weight * orthogonality_loss

                optim.zero_grad()
                loss.backward()
                optim.step()

                interval_rec_loss += (one_weight * reconstruction_loss_one + reconstruction_loss_rest).item()
                interval_kl_loss += (one_weight * kl_loss_one + kl_loss_rest).item()
                interval_orth_loss += orthogonality_loss.item()

                # Log results every `log_interval` steps.
                if count_steps == log_interval:
                    avg_rec = interval_rec_loss / count_steps
                    avg_kl = interval_kl_loss / count_steps
                    avg_orth = interval_orth_loss / count_steps
                    total_loss = avg_rec + kl_weight * avg_kl + orthogonality_weight * avg_orth
                    interval_time = time.time() - interval_start_time

                    with torch.no_grad():
                        logits = categorizer(net.decoder(z_random))

                    class_counts = cl.count_from_logits(logits) / batch_size
                    entropy, margin, ambiguity = cl.ambiguity(logits)
                    entropy, margin, ambiguity = entropy.mean(), margin.mean(), ambiguity.mean()

                    with open(csv_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([global_step, avg_rec, avg_kl, avg_orth,total_loss, interval_time] +\
                                        [entropy.item(), margin.item(), ambiguity.item()] + class_counts.tolist())

                    # Reset interval accumulators.
                    interval_rec_loss = 0.0
                    interval_kl_loss = 0.0
                    interval_orth_loss = 0.0
                    count_steps = 0
                    interval_start_time = time.time()

            # Save a checkpoint at the end of each epoch.
            torch.save(net.state_dict(), f"{checkpoint_dir}/vae_{epoch_i}.pth")

        # Log any remaining steps that didn't complete the final interval.
        if count_steps > 0:
            avg_rec = interval_rec_loss / count_steps
            avg_kl = interval_kl_loss / count_steps
            avg_orth = interval_orth_loss / count_steps
            total_loss = avg_rec + kl_weight * avg_kl + orthogonality_weight * avg_orth
            interval_time = time.time() - interval_start_time

            with torch.no_grad():
                logits = categorizer(net.decoder(z_random))

            class_counts = cl.count_from_logits(logits) / batch_size
            entropy, margin, ambiguity = cl.ambiguity(logits)
            entropy, margin, ambiguity = entropy.mean(), margin.mean(), ambiguity.mean()

            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([global_step, avg_rec, avg_kl, avg_orth, total_loss, interval_time] +\
                                        [entropy.item(), margin.item(), ambiguity.item()] + class_counts.tolist())
















































 