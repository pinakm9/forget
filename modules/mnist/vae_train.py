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


def write_config(model, folder, epochs, batch_size, latent_dim, log_interval='epoch', kl_weight=1., orthogonality_weight=None, one_weight=None):
    sample_dir = f'{folder}/samples'    
    checkpoint_dir = f'{folder}/checkpoints'
    # At the end of the train() function, after training is complete:
    config_data = {
        "training": {
            "epochs": {
                "value": epochs,
                "description": "The number of epochs to train the model."
            },
            "batch_size": {
                "value": batch_size,
                "description": "Number of samples per training batch."
            },
            "learning_rate": {
                "value": 0.001,  # This example uses the default for Adam.
                "description": "Learning rate used by the optimizer."
            },
            "log_interval": {
                "value": log_interval,
                "description": "Logging frequency. 'epoch' logs per epoch or a positive integer logs every fixed number of gradient steps."
            },
            "kl_weight": {
                "value": kl_weight,
                "description": "The weighting factor for the KL divergence loss component."
            },
            "latent_dim": {
                "value": latent_dim,
                "description": "Dimension of the latent space in the VAE."
            },
            "optimizer": {
                "value": "Adam",
                "description": "The type of optimizer used during training."
            }
        },
        "network_setup": {
            "encoder_layers": {
                "value": [784, model.encoder[0].out_features, latent_dim],
                "description": "Encoder configuration: input dimension, hidden layer size, and latent dimension."
            },
            "decoder_layers": {
                "value": [latent_dim, model.decoder[-2].in_features, 784],
                "description": "Decoder configuration: latent dimension, hidden layer size, and output dimension."
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
        }
    }

    if orthogonality_weight is not None:
        config_data["training"]["orthogonality_weight"] = {
            "value": orthogonality_weight,
            "description": "Weight the orthogonality loss."
        }

    if one_weight is not None:
        config_data["training"]["one_weight"] = {
            "value": one_weight,
            "description": "Weight the one loss."
        }

    # Save the configuration to a JSON file in the main folder.
    config_path = f"{folder}/config.json"
    with open(config_path, "w") as config_file:
        json.dump(config_data, config_file, indent=4)



def train(model, folder, epochs, batch_size, latent_dim, device, log_interval='epoch', kl_weight=1.):
    """
    Train a Variational Autoencoder (VAE) model on the MNIST dataset.

    Parameters
    ----------
    model : VAE or str or None
        The VAE model to train. Can be a VAE instance, a path to a saved model 
        state dict, or None to initialize a new model.
    folder : str
        The directory where samples and checkpoints will be saved.
    epochs : int
        The number of epochs to train the model.
    batch_size : int
        The batch size for training.
    latent_dim : int
        The dimension of the latent space.
    device : torch.device
        The device to use for training (e.g., GPU or CPU).
    log_interval : str or int, optional
        If 'epoch' (default), logs data every epoch.
        If an integer > 0, logs data every that many gradient descent steps.
    
    Notes
    -----
    This function trains the VAE by minimizing the combined reconstruction 
    and KL divergence loss. It logs the losses and the time taken into a CSV file 
    either at the end of each epoch or every specified number of gradient descent 
    steps, and saves model checkpoints after every epoch.
    """
    categorizer = cl.get_classifier(device=device)
    sample_dir = f'{folder}/samples'    
    checkpoint_dir = f'{folder}/checkpoints'
    ut.makedirs(sample_dir, checkpoint_dir)
    

    dataloader = datapipe.MNIST().get_dataloader(batch_size)
    if isinstance(model, str):
        net = VAE(latent_dim=latent_dim, device=device).to(device)
        net.load_state_dict(torch.load(model))
    elif model is None:
        net = VAE(latent_dim=latent_dim, device=device).to(device)
    else:
        net = model
    net.train()
    optim = torch.optim.Adam(net.parameters())

    # Determine logging mode and write CSV header accordingly.
    if log_interval == 'epoch':
        header = ["Epoch", "Reconstruction Loss", "KL Loss", "Total Loss", "Time"]
        csv_file = f"{checkpoint_dir}/training_log_epoch.csv"
    elif isinstance(log_interval, int) and log_interval > 0:
        header = ["Step", "Reconstruction Loss", "KL Loss", "Total Loss", "Time"]
        csv_file = f"{checkpoint_dir}/training_log_step.csv"
    else:
        raise ValueError("log_interval must be 'epoch' or a positive integer")
    header += ["Entropy", "Margin", "Ambiguity"] + ["{} Fraction".format(i) for i in range(10)]

    # Write header to CSV file (overwriting any existing file).
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

    write_config(model=net, folder=folder, epochs=epochs, batch_size=batch_size, latent_dim=latent_dim,\
                    log_interval=log_interval, kl_weight=kl_weight)

    z_random = torch.randn((batch_size, latent_dim)).to(device)
    

    if log_interval == 'epoch':
        # logits = torch.zeros((epochs, batch_size, 10)).to(device)
        # Logging per epoch.
        for epoch_i in tqdm(range(1, epochs + 1), desc="Epochs"):
            start_time = time.time()
            epoch_rec_loss = 0.0
            epoch_kl_loss = 0.0
            num_batches = 0

            for _, (real_img, _) in enumerate(dataloader):
                real_img = real_img.view(real_img.shape[0], -1).to(device)

                reconstructed, mu, logvar = net(real_img)
                reconstruction_loss = vl.reconstruction_loss(reconstructed, real_img)
                kl_loss = vl.kl_div(mu, logvar)
                loss = kl_weight*kl_loss + reconstruction_loss

                optim.zero_grad()
                loss.backward()
                optim.step()

                epoch_rec_loss += reconstruction_loss.item()
                epoch_kl_loss += kl_loss.item()
                num_batches += 1
           

            # Compute average losses and elapsed time for the epoch.
            epoch_rec_loss /= num_batches
            epoch_kl_loss /= num_batches
            total_loss = epoch_rec_loss + kl_weight*epoch_kl_loss
            epoch_time = time.time() - start_time

            with torch.no_grad():
                logits = categorizer(net.decoder(z_random))

            class_counts = cl.count_from_logits(logits) / batch_size
            entropy, margin, ambiguity = cl.ambiguity(logits)
            entropy, margin, ambiguity = entropy.mean(), margin.mean(), ambiguity.mean()

            # Log the epoch's results.
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch_i, epoch_rec_loss, epoch_kl_loss, total_loss, epoch_time] +\
                                [entropy.item(), margin.item(), ambiguity.item()] + class_counts.tolist())

            # Save a checkpoint at the end of the epoch.
            torch.save(net.state_dict(), f"{checkpoint_dir}/vae_{epoch_i}.pth")
    else:
        # Logging every fixed number of gradient descent steps.
        global_step = 0
        interval_rec_loss = 0.0
        interval_kl_loss = 0.0
        count_steps = 0
        interval_start_time = time.time()
        logits = torch.zeros((epochs*len(dataloader), batch_size, 10)).to(device)

        for epoch_i in tqdm(range(1, epochs + 1), desc="Epochs"):
            for _, (real_img, _) in enumerate(dataloader):
                global_step += 1
                count_steps += 1
                real_img = real_img.view(real_img.shape[0], -1).to(device)

                reconstructed, mu, logvar = net(real_img)
                reconstruction_loss = vl.reconstruction_loss(reconstructed, real_img)
                kl_loss = vl.kl_div(mu, logvar)
                loss = kl_weight*kl_loss + reconstruction_loss

                optim.zero_grad()
                loss.backward()
                optim.step()

                interval_rec_loss += reconstruction_loss.item()
                interval_kl_loss += kl_loss.item()

                
                
                # Log every 'log_interval' steps.
                if count_steps == log_interval:
                    avg_rec = interval_rec_loss / count_steps
                    avg_kl = interval_kl_loss / count_steps
                    total_loss = avg_rec + kl_weight*avg_kl
                    interval_time = time.time() - interval_start_time

                    with torch.no_grad():
                        logits = categorizer(net.decoder(z_random))

                    class_counts = cl.count_from_logits(logits) / batch_size
                    entropy, margin, ambiguity = cl.ambiguity(logits)
                    entropy, margin, ambiguity = entropy.mean(), margin.mean(), ambiguity.mean()

                    with open(csv_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([global_step, avg_rec, avg_kl, total_loss, interval_time] +\
                                        [entropy.item(), margin.item(), ambiguity.item()] + class_counts.tolist())

                    # Reset interval accumulators.
                    interval_rec_loss = 0.0
                    interval_kl_loss = 0.0
                    count_steps = 0
                    interval_start_time = time.time()

            # Save a checkpoint at the end of each epoch.
            torch.save(net.state_dict(), f"{checkpoint_dir}/vae_{epoch_i}.pth")

        # Log any remaining steps that didn't complete the final interval.
        if count_steps > 0:
            avg_rec = interval_rec_loss / count_steps
            avg_kl = interval_kl_loss / count_steps
            total_loss = avg_rec + avg_kl
            interval_time = time.time() - interval_start_time

            with torch.no_grad():
                logits = categorizer(net.decoder(z_random))
            
            class_counts = cl.count_from_logits(logits) / batch_size
            entropy, margin, ambiguity = cl.ambiguity(logits)
            entropy, margin, ambiguity = entropy.mean(), margin.mean(), ambiguity.mean()
            
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([global_step, avg_rec, avg_kl, total_loss, interval_time] +\
                                        [entropy.item(), margin.item(), ambiguity.item()] + class_counts.tolist())

    # np.save(f"{checkpoint_dir}/logits.npy", logits.cpu().numpy())








