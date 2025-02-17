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



def operate(model, folder, epochs, batch_size, latent_dim, device, log_interval="epoch", kl_weight=1.):
    """
    Train a VAE model and log training metrics either every epoch or every N gradient steps.

    Parameters
    ----------
    model : str or VAE
        Either the path to a saved model (if str) or an instantiated VAE model.
    folder : str
        The directory in which to save samples and checkpoints.
    epochs : int
        The number of epochs to train.
    batch_size : int
        The training batch size.
    latent_dim : int
        The dimensionality of the latent space.
    device : torch.device
        The device to use (CPU, GPU, or MPS).
    log_interval : "epoch" or int, optional
        If "epoch" (default) then logging is done once per epoch.
        If an integer > 0 then logging is performed every that many gradient descent steps.
    kl_weight : float, optional
        Weight of the KL divergence term in the loss.

    """
    categorizer = cl.get_classifier(device=device)
    sample_dir = f'{folder}/samples'    
    checkpoint_dir = f'{folder}/checkpoints'
    ut.makedirs(sample_dir, checkpoint_dir)

    # Determine logging mode and CSV filename/header.
    if log_interval == 'epoch':
        header = ["Epoch", "Reconstruction Loss", "KL Loss", "Orthogonality Loss", "Time"]
        csv_file = f"{checkpoint_dir}/training_log_epoch.csv"
    elif isinstance(log_interval, int) and log_interval > 0:
        header = ["Step", "Reconstruction Loss", "KL Loss", "Orthogonality Loss", "Time"]
        csv_file = f"{checkpoint_dir}/training_log_step.csv"
    else:
        raise ValueError("log_interval must be 'epoch' or a positive integer")
    header += ["Entropy", "Margin", "Ambiguity"] + ["{} Fraction".format(i) for i in range(10)]

    # Write header (overwriting any existing file).
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

    vt.write_config(
        model=model if not isinstance(model, str) else "Loaded from file",
        folder=folder,
        epochs=epochs,
        batch_size=batch_size,
        latent_dim=latent_dim,
        log_interval=log_interval,
        kl_weight=kl_weight
    )

    dataloader_one, dataloader_rest = datapipe.MNIST().get_dataloader_one_vs_rest(batch_size)
    if isinstance(model, str):
        net = VAE(latent_dim=latent_dim, device=device).to(device)
        net.load_state_dict(torch.load(model))
    else:
        net = model

    trainable_params = ut.get_trainable_params(net)
    optim = torch.optim.Adam(trainable_params)
    
    # A random latent vector for computing classifier metrics.
    z_random = torch.randn((batch_size, latent_dim)).to(device)

    # For step-level logging, initialize accumulators.
    if isinstance(log_interval, int):
        global_step = 0
        step_rec_loss = 0.0
        step_kl_loss = 0.0
        step_orth_loss = 0.0
        step_count = 0
        interval_start_time = time.time()

    for epoch_i in tqdm(range(1, epochs + 1), desc="Epochs"):
        start_time = time.time()
        epoch_rec_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_orth_loss = 0.0
        num_batches = 0
        
        for (real_img_one, _), (real_img_rest, _) in zip(dataloader_one, dataloader_rest):
            optim.zero_grad()

            real_img_one = real_img_one.view(real_img_one.shape[0], -1).to(device)
            real_img_rest = real_img_rest.view(real_img_rest.shape[0], -1).to(device)

            # Forward pass on two batches.
            reconstructed_one, mu_one, logvar_one = net(real_img_one)
            reconstructed_rest, mu_rest, logvar_rest = net(real_img_rest)   
            
            reconstruction_loss_one = vl.reconstruction_loss(reconstructed_one, real_img_one)
            reconstruction_loss_rest = vl.reconstruction_loss(reconstructed_rest, real_img_rest)
        
            kl_loss_one = vl.kl_div(mu_one, logvar_one)
            kl_loss_rest = vl.kl_div(mu_rest, logvar_rest)
       
            loss_one = kl_weight * kl_loss_one + reconstruction_loss_one
            loss_rest = kl_weight * kl_loss_rest + reconstruction_loss_rest

            # Compute gradients for each loss.
            g1 = torch.cat([g.view(-1) for g in grad(outputs=loss_one, inputs=trainable_params, retain_graph=True)])
            gr = torch.cat([g.view(-1) for g in grad(outputs=loss_rest, inputs=trainable_params, retain_graph=True)]) 

            # Remove the component of gr that is parallel to g1.
            # (An epsilon is added to avoid division by zero.)
            gr = gr - g1 * (torch.dot(g1, gr)) / (torch.dot(g1, g1) + 1e-8)

            # Reassign gradients to the parameters.
            idx = 0
            for p in trainable_params:
                numel = p.numel()
                p.grad = gr[idx: idx + numel].view(p.shape)
                idx += numel 

            optim.step()

            with torch.no_grad():
                orthogonality_loss = (g1 * gr).mean()**2 
            
            epoch_rec_loss += reconstruction_loss_rest.item()
            epoch_kl_loss += kl_loss_rest.item()
            epoch_orth_loss += orthogonality_loss.item()
            num_batches += 1

            # If using step-level logging, accumulate and log every `log_interval` steps.
            if isinstance(log_interval, int):
                global_step += 1
                step_count += 1
                step_rec_loss += reconstruction_loss_rest.item()
                step_kl_loss += kl_loss_rest.item()
                step_orth_loss += orthogonality_loss.item()

                if step_count == log_interval:
                    interval_time = time.time() - interval_start_time
                    with torch.no_grad():
                        logits = categorizer(net.decoder(z_random))
                    class_counts = cl.count_from_logits(logits) / batch_size
                    entropy, margin, ambiguity = cl.ambiguity(logits)
                    entropy, margin, ambiguity = entropy.mean(), margin.mean(), ambiguity.mean()
                    
                    avg_rec = step_rec_loss / step_count
                    avg_kl = step_kl_loss / step_count
                    avg_orth = step_orth_loss / step_count
                    # (Total loss here is simply the sum of the three averaged losses.)
                    total_loss = avg_rec + avg_kl + avg_orth

                    with open(csv_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([global_step, avg_rec, avg_kl, avg_orth, interval_time] +\
                                        [entropy.item(), margin.item(), ambiguity.item()] + class_counts.tolist())
                    
                    # Reset the step-level accumulators.
                    step_rec_loss = 0.0
                    step_kl_loss = 0.0
                    step_orth_loss = 0.0
                    step_count = 0
                    interval_start_time = time.time()

            # Clean up.
            del g1, gr, orthogonality_loss, loss_one, loss_rest
            gc.collect()

            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.synchronize()
        

        # Log once per epoch if that's the chosen mode.
        if log_interval == "epoch":
             # Compute epoch-level averages.
            epoch_rec_loss /= num_batches
            epoch_kl_loss /= num_batches
            epoch_orth_loss /= num_batches
            epoch_time = time.time() - start_time
            with torch.no_grad():
                logits = categorizer(net.decoder(z_random))
            class_counts = cl.count_from_logits(logits) / batch_size
            entropy, margin, ambiguity = cl.ambiguity(logits)
            entropy, margin, ambiguity = entropy.mean(), margin.mean(), ambiguity.mean()
            
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch_i, epoch_rec_loss, epoch_kl_loss, epoch_orth_loss, epoch_time] +\
                                [entropy.item(), margin.item(), ambiguity.item()] + class_counts.tolist())

        # Save a checkpoint at the end of each epoch.
        torch.save(net.state_dict(), f"{checkpoint_dir}/vae_{epoch_i}.pth")

    # After all epochs, if using step-level logging and there are remaining steps, log them.
    if isinstance(log_interval, int) and step_count > 0:
        interval_time = time.time() - interval_start_time
        with torch.no_grad():
            logits = categorizer(net.decoder(z_random))
        class_counts = cl.count_from_logits(logits) / batch_size
        entropy, margin, ambiguity = cl.ambiguity(logits)
        entropy, margin, ambiguity = entropy.mean(), margin.mean(), ambiguity.mean()
        
        avg_rec = step_rec_loss / step_count
        avg_kl = step_kl_loss / step_count
        avg_orth = step_orth_loss / step_count
        total_loss = avg_rec + avg_kl + avg_orth

        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([global_step, avg_rec, avg_kl, avg_orth, interval_time] +\
                            [entropy.item(), margin.item(), ambiguity.item()] + class_counts.tolist())
