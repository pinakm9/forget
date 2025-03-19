import torch 
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os, sys, csv
import time
import json
from torch.autograd import grad

sys.path.append(os.path.abspath('../modules'))
import utility as ut
import datapipe
import vae_loss as vl
from vae import VAE
import classifier as cl
import vae_viz as viz


def write_config(model, folder, epochs, epoch_length, batch_size, latent_dim, collect_interval='epoch', log_interval='epoch', kl_weight=1., orthogonality_weight=None,\
                uniformity_weight=None, forget_weight=None, all_digits=None, forget_digit=None, img_ext='jpg'):
    sample_dir = f'{folder}/samples'    
    checkpoint_dir = f'{folder}/checkpoints'
    # At the end of the train() function, after training is complete:
    config_data = {
        "training": {
            "epochs": {
                "value": epochs,
                "description": "The number of epochs to train the model."
            },
            "epoch_length": {
                "value": epoch_length,
                "description": "The number of descent steps in an epoch."
            },
            "batch_size": {
                "value": batch_size,
                "description": "Number of samples per training batch."
            },
            "learning_rate": {
                "value": 0.001,  # This example uses the default for Adam.
                "description": "Learning rate used by the optimizer."
            },
            "kl_weight": {
                "value": kl_weight,
                "description": "The weighting factor for the KL divergence loss component."
            },
            "optimizer": {
                "value": "Adam",
                "description": "The type of optimizer used during training."
            },
            "device": { 
                "value": str(model.device),
                "description": "The device used for training."
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
        },
        "experiment": {
            "latent_dim": {
                "value": latent_dim,
                "description": "Dimension of the latent space of the VAE."
            },
            "img_ext": {
                "value": img_ext,
                "description": "File extension for sample images."
            },
            "collect_interval": {
                "value": collect_interval,
                "description": "Frequency of collecting samples."
            },
            "log_interval": {
                "value": log_interval,
                "description": "Frequency of logging training loss."
            }
            
        }
    }

    if orthogonality_weight is not None:
        config_data["training"]["orthogonality_weight"] = {
            "value": orthogonality_weight,
            "description": "Weight of the orthogonality loss."
        }

    if forget_weight is not None:
        config_data["training"]["forget_weight"] = {
            "value": forget_weight,
            "description": "Weight of the forget loss."
        }

    if uniformity_weight is not None:
        config_data["training"]["uniformity_weight"] = {
            "value": uniformity_weight,
            "description": "Weight of the uniformity loss."
        }

    if forget_digit is not None:
        config_data["experiment"]["forget_digit"] = {
            "value": forget_digit,
            "description": "Digit to forget."
        }

    if len(all_digits) == 10:
        config_data["experiment"]["dataset"] = {
            "value": 'MNIST-' + ''.join(list(map(str, all_digits))),
            "description": "Original dataset."
        }
    else:
        config_data["experiment"]["dataset"] = {
            "value": 'MNIST',
            "description": "Original dataset."
        }

    # Save the configuration to a JSON file in the main folder.
    config_path = f"{folder}/config.json"
    with open(config_path, "w") as config_file:
        json.dump(config_data, config_file, indent=4)



def init_model(model, latent_dim, device):
    if isinstance(model, str):
        try:
            net = torch.load(model)
        except:
            net = VAE(latent_dim=latent_dim, device=device)
            net.load_state_dict(torch.load(model))
            net.to(device)
    elif model is None:
        net = VAE(latent_dim=latent_dim, device=device)
        net.to(device)
    else:
        net = model
    net.train()
    return net


def get_processor(net, identifier, z_random, weights, optim, all_digits, forget_digit):
    digits = all_digits
    def process_batch(real_img):
        """
        Perform a forward + backward pass on a single batch, returning the individual loss terms.
        """
        kl_weight, uniformity_weight = weights
        real_img = real_img.view(real_img.shape[0], -1).to(net.device)
        # Forward pass
        start_time =  time.time()
        reconstructed_img, mu, logvar = net(real_img)
        generated_img = net.decoder(z_random)
        logits = identifier(generated_img)

        # Compute losses
        reconstruction_loss = vl.reconstruction_loss(reconstructed_img, real_img)
        kl_loss = vl.kl_div(mu, logvar)
        uniformity_loss = vl.uniformity_loss(logits, digits)
        
        # Combine into total loss
        loss = reconstruction_loss + kl_weight * kl_loss + uniformity_weight * uniformity_loss

        # Backprop + optimize
        optim.zero_grad()
        loss.backward()
        optim.step()
        elapsed_time = time.time() - start_time

        return reconstruction_loss.item(), kl_loss.item(), uniformity_loss.item(), generated_img, logits, elapsed_time
    return process_batch


def get_logger(net, identifier, dataloader, weights, csv_file, log_interval, all_digits):
    def log_results(step, losses, elapsed_time, real_img, generated_img, logits):
        """
        Log a single row of results (rec, KL, uniformity, total loss, time, and class stats) to the CSV file.
        """
        if step % log_interval == 0:
            kl_weight, uniformity_weight = weights
            retain_sample, _ = next(iter(dataloader['retain']))
            forget_sample, _ = next(iter(dataloader['forget']))
            retain_sample = retain_sample.view(retain_sample.shape[0], -1).to(net.device)
            forget_sample = forget_sample.view(forget_sample.shape[0], -1).to(net.device)
            orthogonality = vl.orthogonality_loss(net, identifier, retain_sample, forget_sample, kl_weight, uniformity_weight, all_digits)
            losses.insert(-2, orthogonality.item())

            img_quality = [cl.frechet_inception_distance(real_img, generated_img, identifier), cl.inception_score(logits).item()]
            # Compute class counts and ambiguities
            class_counts = cl.count_from_logits(logits) / logits.shape[0]
            entropy, margin = cl.ambiguity(logits)
            img_quality += [entropy.mean().item(), margin.mean().item()]

            # Write row to the CSV file
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([step] + losses + [elapsed_time] + img_quality + class_counts.tolist())
    return log_results


def get_saver(net, save_steps, checkpoint_dir, epoch_length):
    def save(step):
        if step in save_steps:
            if step % epoch_length == 0:
                torch.save(net, f"{checkpoint_dir}/vae_epoch_{int(step/epoch_length)}.pth")
            else:
                torch.save(net, f"{checkpoint_dir}/vae_step_{step}.pth")
    return save


def get_collector(sample_dir, collect_interval, grid_size, img_ext='jpg'):
    def collect_samples(generated_img, step):
        if step % collect_interval == 0:
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
            axes = axes.flatten()
            generated_img = generated_img.detach().cpu().reshape(-1, 28, 28)
            for i, ax in enumerate(axes):
                ax.imshow(generated_img[i], cmap="gray", vmin=0, vmax=1)
                ax.axis("off")
            fig.suptitle(f"Step = {step}")
            plt.savefig(f"{sample_dir}/sample_{step}.{img_ext}", bbox_inches="tight", dpi=300)
            plt.close(fig)
    return collect_samples


def init(model='./vae.pth', folder='.', num_steps=100, batch_size=100, latent_dim=2, save_steps=None, collect_interval='epoch', log_interval=10, kl_weight=1.,\
        uniformity_weight=1e4, orthogonality_weight=10., forget_weight=None, all_digits=None, forget_digit=None, img_ext='jpg',\
        classifier_path="../data/MNIST/classifiers/MNISTClassifier.pth", train_mode='original'):
    device = ut.get_device()
    identifier = cl.get_classifier(classifier_path, device=device)
    sample_dir = f'{folder}/samples'
    checkpoint_dir = f'{folder}/checkpoints'
    ut.makedirs(sample_dir, checkpoint_dir)
    dataloader = {'original': datapipe.MNIST().get_dataloader(batch_size, all_digits=all_digits)}
    dataloader_retain, dataloader_forget = datapipe.MNIST().get_dataloader_rf(batch_size, all_digits=all_digits, forget_digit=forget_digit)
    dataloader['retain'] = dataloader_retain
    dataloader['forget'] = dataloader_forget
    net = init_model(model, latent_dim, device)
    optim = torch.optim.Adam(net.parameters())
    z_random = torch.randn((batch_size, latent_dim)).to(device)
    epoch_length = len(dataloader['original']) if train_mode == 'original' else min(len(dataloader['forget']), len(dataloader['retain']))
    epochs = int(np.ceil(num_steps/epoch_length))
    num_steps = epochs * epoch_length
    grid_size = int(np.ceil(np.sqrt(batch_size)))

    if isinstance(save_steps, list):
        save_steps = save_steps + [epoch_length, num_steps]
        save_steps = list(set(save_steps))
        save_steps.sort()
    else:
        save_steps = [epoch_length, num_steps]

    if collect_interval == 'epoch':
        collect_interval = epoch_length

    if log_interval == 'epoch':         
        log_interval = epoch_length

    # ---------------------------------------------------
    # Prepare CSV logging
    # ---------------------------------------------------     
    csv_file = f"{checkpoint_dir}/training_log.csv"
    # Extra columns for class distribution + ambiguity
    header = ["Step", "Reconstruction Loss", "KL Loss", "Uniformity Loss", "Orthogonality Loss", "Total Loss", "Time"]
    header += ["FID", "IS", "Entropy", "Margin"] + [f"{i} Fraction" for i in range(10)]
    # Write CSV header
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
    # Save config
    write_config(model=net, folder=folder, epochs=epochs, epoch_length=epoch_length, batch_size=batch_size, latent_dim=latent_dim,\
                collect_interval=collect_interval, log_interval=log_interval, kl_weight=kl_weight, uniformity_weight=uniformity_weight,\
                orthogonality_weight=orthogonality_weight, forget_weight=forget_weight, all_digits=all_digits, forget_digit=forget_digit,\
                img_ext=img_ext)

    return net, dataloader, optim, z_random, identifier, sample_dir, checkpoint_dir, epoch_length, epochs,\
           num_steps,save_steps, collect_interval, log_interval, csv_file, device, grid_size



def train(model, folder, num_steps, batch_size, latent_dim=2, save_steps=None, collect_interval='epoch', log_interval=10,\
          kl_weight=1., uniformity_weight=1e4, all_digits=list(range(10)), forget_digit=1,\
          img_ext='jpg', classifier_path="../data/MNIST/classifiers/MNISTClassifier.pth"):
    # ---------------------------------------------------
    # Setup
    # ---------------------------------------------------
    net, dataloader, optim, z_random, identifier, sample_dir, checkpoint_dir, epoch_length, epochs,\
    num_steps, save_steps, collect_interval, log_interval, csv_file, device, grid_size \
    = init(model, folder, num_steps, batch_size, latent_dim=latent_dim, save_steps=save_steps, collect_interval=collect_interval,\
           log_interval=log_interval, kl_weight=kl_weight, uniformity_weight=uniformity_weight, orthogonality_weight=0.,\
           all_digits=all_digits, forget_digit=forget_digit, img_ext=img_ext, classifier_path=classifier_path)
    process_batch = get_processor(net, identifier, z_random, (kl_weight, uniformity_weight), optim, all_digits, forget_digit)    
    log_results = get_logger(net, identifier, dataloader, (kl_weight, uniformity_weight), csv_file, log_interval, all_digits)
    save = get_saver(net, save_steps, checkpoint_dir, epoch_length)
    collect_samples = get_collector(sample_dir, collect_interval, grid_size, img_ext)   

    # ---------------------------------------------------
    # Main training loop
    # ---------------------------------------------------
    global_step = 0
    for _ in tqdm(range(1, epochs + 1), desc="Epochs"):
        for _, (real_img, _) in enumerate(dataloader['original']):
            global_step += 1
            # -- Process a single batch
            rec_loss, kl_loss, unif_loss, generated_img, logits, elapsed_time = process_batch(real_img)
            loss = rec_loss + kl_weight * kl_loss + uniformity_weight * unif_loss
            log_results(step=global_step, losses=[rec_loss, kl_loss, unif_loss, loss], elapsed_time=elapsed_time, real_img=real_img, generated_img=generated_img, logits=logits)
            save(step=global_step)
            collect_samples(generated_img, step=global_step)
    viz.summarize_training(folder)
