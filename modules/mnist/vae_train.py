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
    """
    Writes the configuration of the VAE experiment to a JSON file.

    Parameters:
    model (nn.Module): The VAE model.
    folder (str): The root directory where the samples and checkpoints will be saved.
    epochs (int): The number of epochs to train the model.
    epoch_length (int): The number of descent steps in an epoch.
    batch_size (int): The number of samples per training batch.
    latent_dim (int): The dimensionality of the latent space.
    collect_interval (int or str): The interval at which to collect samples. If 'epoch', this is set to the epoch length.
    log_interval (int or str): The interval at which to log metrics. If 'epoch', this is set to the epoch length.
    kl_weight (float): The weight for the KL loss term.
    orthogonality_weight (float): The weight for the orthogonality loss term. If None, the orthogonality loss is not used.
    uniformity_weight (float): The weight for the uniformity loss term. If None, the uniformity loss is not used.
    forget_weight (float): The weight for the forget loss term. If None, the forget loss is not used.
    all_digits (list): The list of all digits to use. If None, all digits are used.
    forget_digit (int): The digit to forget. If None, no digit is forgotten.
    img_ext (str): The file extension for the saved images.

    """
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
    """
    Initialize a VAE model based on the provided input or load a pre-trained model.

    Parameters
    ----------
    model : str, VAE, or None
        If a string, it is treated as a file path to load a pre-trained model.
        If a VAE instance, it is used directly.
        If None, a new VAE model is instantiated with the specified latent dimension.
    latent_dim : int
        The dimensionality of the latent space for a new VAE model.
    device : str
        The device to which the model should be moved, e.g., 'cuda' or 'cpu'.

    Returns
    -------
    VAE
        The initialized VAE model moved to the specified device.
    """

    if isinstance(model, str):
        try:
            net = torch.load(model, weights_only=False, map_location=device)
            net.to(device)
        except:
            net = VAE(latent_dim=latent_dim, device=device)
            net.load_state_dict(torch.load(model, weights_only=True, map_location=device))
    
    elif model is None:
        net = VAE(latent_dim=latent_dim, device=device)
    else:
        net = model
    net.to(device)
    # net.train()
    return net


def get_processor(net, identifier, z_random, weights, optim, all_digits, forget_digit):
    """
    Returns a function that processes a batch of images through a VAE network and computes the necessary gradients.

    The function performs a forward and backward pass on a batch of images, calculating the reconstruction loss, 
    KL divergence, and a uniformity loss. It then adjusts the gradients to ensure orthogonality, performs an 
    optimization step, and returns the relevant metrics.

    Parameters:
        net (nn.Module): The VAE model.
        identifier (nn.Module): The model used for logits computation.
        z_random (torch.tensor): Random latent codes for the decoder.
        weights (tuple): Contains weights for KL divergence and uniformity loss.
        optim (torch.optim.Optimizer): Optimizer for the VAE.
        all_digits (list): List of all class labels.
        forget_digit (int): The class label to forget.

    Returns:
        function: A function that takes a batch of images and returns the reconstruction loss, 
                  KL divergence, uniformity loss, generated image, logits, and elapsed time.
    """
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
    """
    Returns a function that logs training results to a CSV file at specified intervals.

    Parameters:
    net: The VAE model.
    identifier: The identifier model.
    dataloader: A dictionary of two data loaders, one for retain and one for forget classes.
    weights: A list of length 2 containing the weights of the KL divergence and uniformity loss.
    csv_file: The path to the CSV file where results are logged.
    log_interval: The interval at which to log the results.
    all_digits: A list of all digit labels.

    Returns:
    A function that logs a row of results, including reconstruction loss, KL divergence,
    uniformity, total loss, elapsed time, image quality metrics, and class statistics, to the CSV file.
    """
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
            losses.insert(-1, orthogonality.item())

            img_quality = [] # [cl.frechet_inception_distance(real_img, generated_img, identifier), cl.inception_score(logits).item()]
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
    """
    Returns a function that saves the VAE model to a file at the given step.
    
    Saves the model to a file named vae_epoch_<epoch number>.pth if step is a multiple of epoch_length,
    otherwise saves to a file named vae_step_<step number>.pth.

    Parameters:
    net (torch.nn.Module): The VAE model.
    save_steps (list): The steps at which to save the model.
    checkpoint_dir (str): The directory to save the model to.
    epoch_length (int): The length of an epoch in terms of steps.
    """
    def save(step):
        if step in save_steps:
            if step % epoch_length == 0:
                torch.save(net, f"{checkpoint_dir}/vae_epoch_{int(step/epoch_length)}.pth")
            else:
                torch.save(net, f"{checkpoint_dir}/vae_step_{step}.pth")
    return save


def get_collector(sample_dir, collect_interval, grid_size, img_ext='jpg'):
    """
    Creates a function to save generated images at specified intervals.

    The returned function saves a grid of generated images to the specified directory 
    when the current step is a multiple of the collect interval. Images are saved in 
    grayscale with a resolution of 28x28 pixels.

    Parameters:
    sample_dir (str): Directory to save the images.
    collect_interval (int): Interval at which images are saved.
    grid_size (int): Size of the grid for arranging images.
    img_ext (str): File extension for saved images, default is 'jpg'.

    Returns:
    function: A function that takes a generated image tensor and step number, saving 
    the image grid if the step condition is met.
    """

    def collect_samples(generated_img, step):
        if step % collect_interval == 0:
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
            axes = axes.flatten()
            generated_img = generated_img.detach().cpu().reshape(-1, 28, 28)
            for i, ax in enumerate(axes):
                ax.imshow(generated_img[i], cmap="gray", vmin=0, vmax=1)
                ax.axis("off")
            # fig.suptitle(f"Step = {step}")
            plt.savefig(f"{sample_dir}/sample_{step}.{img_ext}", bbox_inches="tight", dpi=300)
            plt.close(fig)
    return collect_samples


def init(model='./vae.pth', folder='.', num_steps=100, batch_size=100, latent_dim=2, save_steps=None, collect_interval='epoch', log_interval=10, kl_weight=1.,\
        uniformity_weight=1e4, orthogonality_weight=10., forget_weight=None, all_digits=None, forget_digit=None, img_ext='jpg',\
        classifier_path="../data/MNIST/classifiers/MNISTClassifier.pth", train_mode='original', data_path='../../data/MNIST'):
    """
    Initializes all components for VAE training on MNIST digits.

    Parameters
    ----------
    model : str or torch.nn.Module
        Path to a saved model or a model itself.
    folder : str
        Root directory where the samples and checkpoints will be saved. Default is '.'.
    num_steps : int
        Number of training steps. Default is 100.
    batch_size : int
        Batch size for training. Default is 100.
    latent_dim : int
        Dimensionality of the latent space. Default is 2.
    save_steps : list or str or None
        Steps at which to save the model. If list, saves at the specified steps. If str, must be 'epoch', which saves at the end of each epoch. If None, never saves.
    collect_interval : str or int
        Interval at which to collect samples. If str, must be 'epoch', which collects at the end of each epoch. If int, collects at the specified step interval. Default is 'epoch'.
    log_interval : str or int
        Interval at which to log results. If str, must be 'epoch', which logs at the end of each epoch. If int, logs at the specified step interval. Default is 10.
    kl_weight : float
        Weight for the KL loss term. Default is 1.
    uniformity_weight : float
        Weight for the uniformity loss term. Default is 1e4.
    orthogonality_weight : float
        Weight for the orthogonality loss term. Default is 10.
    forget_weight : float or None
        Weight for the forget loss term. If None, the forget loss is not used. Default is None.
    all_digits : list or None
        List of all digits to use. If None, all digits are used. Default is None.
    forget_digit : int or None
        Digit to forget. If None, no digit is forgotten. Default is None.
    img_ext : str
        File extension for saved images. Default is 'jpg'.
    classifier_path : str
        Path to the classifier model. Default is "../data/MNIST/classifiers/MNISTClassifier.pth".
    train_mode : str
        Training mode. Options are 'original', 'retain', or 'forget'. Default is 'original'.
    data_path : str
        Path to the MNIST dataset. Default is '../../data/MNIST'.

    Returns
    -------
    net : torch.nn.Module
        The VAE model.
    dataloader : dict
        A dictionary containing the dataloaders for the retain and forget datasets.
    optim : torch.optim.Optimizer
        The optimizer for the VAE model.
    z_random : torch.Tensor
        A tensor of random noise for generating samples.
    identifier : torch.nn.Module
        The classifier model.
    sample_dir : str
        The directory where the samples will be saved.
    checkpoint_dir : str
        The directory where the model checkpoints will be saved.
    epoch_length : int
        The length of an epoch in terms of training steps.
    epochs : int
        The number of epochs to train for.
    num_steps : int
        The total number of training steps.
    save_steps : list
        The steps at which to save the model.
    collect_interval : int
        The interval at which to collect samples.
    log_interval : int
        The interval at which to log results.
    csv_file : str
        The path to the CSV file where the results will be logged.
    device : torch.device
        The device on which the VAE model is running.
    grid_size : int
        The size of the grid for arranging generated images.
    """
    device = ut.get_device()
    identifier = cl.get_classifier(classifier_path, device=device)
    sample_dir = f'{folder}/samples'
    checkpoint_dir = f'{folder}/checkpoints'
    ut.makedirs(sample_dir, checkpoint_dir)
    dataloader = {'original': datapipe.MNIST().get_dataloader(batch_size, root=data_path, all_digits=all_digits)}
    dataloader_retain, dataloader_forget = datapipe.MNIST().get_dataloader_rf(batch_size, root=data_path, all_digits=all_digits, forget_digit=forget_digit)
    dataloader['retain'] = dataloader_retain
    dataloader['forget'] = dataloader_forget
    net = init_model(model, latent_dim, device)
    optim = torch.optim.Adam(net.parameters())
    z_random = torch.randn((batch_size, latent_dim)).to(device)
    epoch_length = len(dataloader['original']) if train_mode == 'original' else min(len(dataloader['forget']), len(dataloader['retain']))
    epochs = int(np.ceil(num_steps/epoch_length))
    num_steps = epochs * epoch_length
    grid_size = 8#int(np.ceil(np.sqrt(batch_size)))

    if isinstance(save_steps, list):
        save_steps = save_steps + [epoch_length, num_steps]
        save_steps = list(set(save_steps))
        save_steps.sort()
    elif save_steps == "epoch":
        save_steps = list(range(epoch_length, num_steps + 1, epoch_length))
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
    header += ["Entropy", "Margin"] + [f"{i} Fraction" for i in range(10)]
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
          img_ext='jpg', classifier_path="../data/MNIST/classifiers/MNISTClassifier.pth", data_path='../../data/MNIST'):
    """
    Train a VAE on MNIST digits, with a custom loop to alternate between ascent and descent steps.

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
    data_path : str, optional
        Path to the MNIST dataset. Defaults to '../../data/MNIST'.
    """
    # ---------------------------------------------------
    # Setup
    # ---------------------------------------------------
    net, dataloader, optim, z_random, identifier, sample_dir, checkpoint_dir, epoch_length, epochs,\
    num_steps, save_steps, collect_interval, log_interval, csv_file, device, grid_size \
    = init(model, folder, num_steps, batch_size, latent_dim=latent_dim, save_steps=save_steps, collect_interval=collect_interval,\
           log_interval=log_interval, kl_weight=kl_weight, uniformity_weight=uniformity_weight, orthogonality_weight=0.,\
           all_digits=all_digits, forget_digit=forget_digit, img_ext=img_ext, classifier_path=classifier_path, train_mode='original', data_path=data_path)
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
