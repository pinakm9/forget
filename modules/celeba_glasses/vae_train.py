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
                uniformity_weight=None, forget_weight=None, all_classes=None, forget_class=None, img_ext='jpg'):
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
                "value": str(next(model.parameters()).device),
                "description": "The device used for training."
            }
        },
        # "network_setup": {
        #     "encoder_layers": {
        #         "value": [784, model.encoder[0].out_features, latent_dim],
        #         "description": "Encoder configuration: input dimension, hidden layer size, and latent dimension."
        #     },
        #     "decoder_layers": {
        #         "value": [latent_dim, model.decoder[-2].in_features, 784],
        #         "description": "Decoder configuration: latent dimension, hidden layer size, and output dimension."
        #     }
        # },
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

    if forget_class is not None:
        config_data["experiment"]["forget_class"] = {
            "value": forget_class,
            "description": "class to forget."
        }

    if len(all_classes) == 10:
        config_data["experiment"]["dataset"] = {
            "value": 'MNIST-' + ''.join(list(map(str, all_classes))),
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
    Initialize and return a VAE model.

    Parameters
    ----------
    model : str or VAE or None
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
        The initialized VAE model in training mode.
    """

    if isinstance(model, str):
        try:
            net = torch.load(model, weights_only=False, map_location=device)
            net.to(device) #<--- make sure net is not an OrderedDict
        except:
            net = VAE(latent_dim=latent_dim)
            net.load_state_dict(torch.load(model, weights_only=True, map_location=device))
    elif model is None:
        net = VAE(latent_dim=latent_dim)
    else:
        net = model
    net.to(device)
    return net



def get_processor(net, identifier, z_random, weights, optim, all_classes, forget_class):
    """
    Returns a function that performs a forward + backward pass on a single batch of images from the generator.
    
    Parameters:
    net (nn.Module): the VAE model
    identifier (nn.Module): the identifier model
    z_random (torch.tensor): a tensor of random latent codes
    weights (list): a list of length 2 containing the weights of the KL divergence and uniformity loss
    optim (torch.optim.Optimizer): the optimizer for the VAE
    all_classes (list): a list of all class labels
    forget_class (int): the class label to forget
    
    Returns:
    process_batch (function): a function that takes a batch of images and returns the reconstruction loss, KL divergence, uniformity loss, generated image, logits of the identifier, and elapsed time.
    """
    classes = all_classes
    # device = next(net.parameters()).device
    # @ut.timer
    def process_batch(real_img):
        """
        Perform a forward + backward pass on a single batch, returning the individual loss terms.
        """
        kl_weight, uniformity_weight = weights
        # Forward pass
        start_time =  time.time()
        # net.eval()
        generated_img = net.decode(z_random)

        # net.train()
        reconstructed_img, mu, logvar = net(real_img)
        logits = identifier(generated_img)

        # Compute losses
        reconstruction_loss = vl.reconstruction_loss(reconstructed_img, real_img)
        kl_loss = vl.kl_div(mu, logvar)
        uniformity_loss = vl.uniformity_loss(logits, classes)
        
        # Combine into total loss
        loss = reconstruction_loss + kl_weight * kl_loss + uniformity_weight * uniformity_loss

        # Backprop + optimize
        optim.zero_grad()
        loss.backward()
        optim.step()
        elapsed_time = time.time() - start_time

        return reconstruction_loss.item(), kl_loss.item(), uniformity_loss.item(), generated_img, logits, elapsed_time
    return process_batch



def get_logger(net, identifier, dataloader, weights, csv_file, log_interval, all_classes):
    """
    Return a function that logs a single row of results (rec, KL, uniformity, total loss, time, and class stats)
    to the CSV file.

    Parameters:
    net (nn.Module): The VAE model.
    identifier (nn.Module): The image classifier.
    dataloader (dict): Data loader for retain and forget classes.
    weights (list): List of KL, uniformity, and forget weights.
    csv_file (str): Path to the CSV file.
    log_interval (int): Interval at which to log.
    all_classes (list): List of all classes.

    Returns:
    A function that takes in step, losses, elapsed_time, real_img, generated_img, and logits, and logs them to the CSV file.
    """
    device = next(net.parameters()).device
    # @ut.timer
    def log_results(step, losses, elapsed_time, real_img, generated_img, logits):
        """
        Log a single row of results (rec, KL, uniformity, total loss, time, and class stats) to the CSV file.
        """
        if step % log_interval == 0:
            kl_weight, uniformity_weight = weights
            retain_sample, _ = next(iter(dataloader['retain']))
            forget_sample, _ = next(iter(dataloader['forget']))
            retain_sample = retain_sample.to(device)
            forget_sample = forget_sample.to(device)
            orthogonality = vl.orthogonality_loss(net, identifier, retain_sample, forget_sample, kl_weight, uniformity_weight, all_classes)
            losses.insert(-1, orthogonality.item())

            img_quality = [] #[cl.frechet_inception_distance(real_img, generated_img, identifier), cl.inception_score(logits).item()]
            # Compute class counts and ambiguities
            class_counts = cl.count_from_logits(logits) / logits.shape[0]
            entropy, margin = cl.ambiguity(logits)
            img_quality += [entropy.mean().item(), margin.mean().item()]

            # Write row to the CSV file
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([step] + losses + [elapsed_time] + img_quality + [1 - class_counts, class_counts])
    return log_results



def get_saver(net, save_steps, checkpoint_dir, epoch_length):
    def save(step):
        """
        Saves the VAE model to a file at the given step.

        Saves the model to a file named vae_epoch_<epoch number>.pth if step is a multiple of epoch_length,
        otherwise saves to a file named vae_step_<step number>.pth.

        Parameters:
        step (int): The current step number.
        """
        if step in save_steps:
            if step % epoch_length == 0:
                torch.save(net, f"{checkpoint_dir}/vae_epoch_{int(step/epoch_length)}.pth")
            else:
                torch.save(net, f"{checkpoint_dir}/vae_step_{step}.pth")
    return save


def get_collector(sample_dir, collect_interval, grid_size, img_ext='jpg'):
    """
    Returns a function that takes a generated image and a step number, and saves the
    image to a file in the given sample directory if the step number is a multiple of
    the given collect_interval. The image is saved as a <img_ext> file with the name
    "sample_<step number>.<img_ext>".

    The generated image is expected to be a 4D tensor with shape (batch_size, 3, H, W).
    The image is saved with a grid size of grid_size x grid_size, and the pixels are
    normalized to the range [0, 1].

    Parameters:
    sample_dir (str): The directory where the images will be saved.
    collect_interval (int): The interval at which images will be saved.
    grid_size (int): The size of the grid on which the images will be arranged.
    img_ext (str): The file extension for the saved images. Default is 'jpg'.
    """
    # @ut.timer
    def collect_samples(generated_img, step):
        generated_img = generated_img.detach().cpu()
        # if step == num_steps:
        #     np.save(f"{sample_dir}/sample_{step}.npy", generated_img.numpy())
        if step % collect_interval == 0:
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
            axes = axes.flatten()
            generated_img = generated_img[:grid_size*grid_size].clamp(0, 1)
            generated_img = generated_img.permute(0, 2, 3, 1).numpy()  # BxCxHxW -> BxHxWxC

            for i, ax in enumerate(axes):
                if i < generated_img.shape[0]:
                    ax.imshow(generated_img[i])
                    ax.axis("off")

            plt.savefig(f"{sample_dir}/sample_{step}.{img_ext}", bbox_inches="tight", dpi=300)
            plt.close(fig)

    return collect_samples



def init(model='../../data/CelebA/vae/vae_200.pth', folder='.', num_steps=100, batch_size=100, latent_dim=512, save_steps=None, collect_interval='epoch', log_interval=10, kl_weight=1.,\
        uniformity_weight=1e4, orthogonality_weight=10., forget_weight=None, all_classes=None, forget_class=None, img_ext='jpg',\
        classifier_path="../../data/CelebA/cnn/cnn_10.pth", train_mode='original', data_path='../../data/CelebA/dataset', max_data=None):
    
    """
    Initializes a VAE model and its associated components for training.

    Parameters:
    model (str): The path to the model to be loaded. Default is '../../data/CelebA/vae/vae_200.pth'.
    folder (str): The root directory where the samples and checkpoints will be saved. Default is '.'.
    num_steps (int): The total number of training steps. Default is 100.
    batch_size (int): The batch size for training. Default is 100.
    latent_dim (int): The dimensionality of the latent space. Default is 2.
    save_steps (list): A list of steps at which to save the model. Default is [100].
    collect_interval (int or str): The interval at which to collect samples. If 'epoch', this is set to the epoch length. Default is 'epoch'.
    log_interval (int or str): The interval at which to log metrics. If 'epoch', this is set to the epoch length. Default is 10.
    kl_weight (float): The weight for the KL loss term. Default is 1.
    uniformity_weight (float): The weight for the uniformity loss term. Default is 1e4.
    orthogonality_weight (float): The weight for the orthogonality loss term. Default is 10.
    forget_weight (float or None): The weight for the forget loss term. If None, the forget loss is not used. Default is None.
    all_classes (list or None): The list of classes to be used for training. If None, all classes are used. Default is None.
    forget_class (int or None): The class to be forgotten. If None, no class is forgotten. Default is None.
    img_ext (str): The file extension for the saved images. Default is 'jpg'.
    classifier_path (str): The path to the classifier model. Default is "../../data/CelebA/cnn/cnn_10.pth".
    train_mode (str): The training mode. Options are 'original', 'retain', or 'forget'. Default is 'original'.
    data_path (str): The path to the data folder. Defaults to'../../data/CelebA/dataset'

    Returns:
    net (nn.Module): The VAE model.
    dataloader (dict): A dictionary containing the data loaders for the different modes.
    optim (torch.optim.Optimizer): The optimizer for training.
    z_random (torch.Tensor): A tensor of random latent codes.
    identifier (nn.Module): The classifier model.
    sample_dir (str): The directory where the samples will be saved.
    checkpoint_dir (str): The directory where the model checkpoints will be saved.
    epoch_length (int): The length of an epoch in terms of steps.
    epochs (int): The total number of epochs.
    num_steps (int): The total number of training steps.
    save_steps (list): A list of steps at which to save the model.
    collect_interval (int): The interval at which to collect samples.
    log_interval (int): The interval at which to log metrics.
    csv_file (str): The path to the CSV file for logging.
    device (torch.device): The device on which to train the model.
    grid_size (int): The size of the grid on which the images will be arranged.
    """
    device = ut.get_device()
    identifier = cl.get_classifier(classifier_path, device=device)
    sample_dir = f'{folder}/samples'
    checkpoint_dir = f'{folder}/checkpoints'
    ut.makedirs(sample_dir, checkpoint_dir)
    dataloader = {'original': datapipe.CelebAData(img_path=data_path, max_size=max_data).get_dataloader(batch_size)}
    dataloader_retain, dataloader_forget = datapipe.CelebAAttrFRData(img_path=data_path, attr_path=f'{data_path}/list_attr_celeba.csv', max_size=max_data).get_dataloader_rf(batch_size)
    dataloader['retain'] = dataloader_retain
    dataloader['forget'] = dataloader_forget
    net = init_model(model, latent_dim, device)
    optim = torch.optim.Adam(net.parameters(), lr=1e-3)
    z_random = torch.randn((batch_size, latent_dim)).to(device)
    epoch_length = len(dataloader['original']) if train_mode == 'original' else min(len(dataloader['forget']), len(dataloader['retain']))
    epochs = int(np.ceil(num_steps/epoch_length))
    num_steps = epochs * epoch_length
    grid_size = 5#int(np.ceil(np.sqrt(batch_size)))

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
    # header = ["Step", "Reconstruction Loss", "KL Loss", "Uniformity Loss", "Orthogonality Loss", "Total Loss", "Time"]
    # header += ["FID", "IS", "Entropy", "Margin"] + [f"{i} Fraction" for i in range(2)]

    header = ["Step", "Reconstruction Loss", "KL Loss", "Uniformity Loss", "Orthogonality Loss", "Total Loss", "Time"]
    header += ["Entropy", "Margin"] + [f"{i} Fraction" for i in range(2)]
    # Write CSV header
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
    # Save config
    write_config(model=net, folder=folder, epochs=epochs, epoch_length=epoch_length, batch_size=batch_size, latent_dim=latent_dim,\
                collect_interval=collect_interval, log_interval=log_interval, kl_weight=kl_weight, uniformity_weight=uniformity_weight,\
                orthogonality_weight=orthogonality_weight, forget_weight=forget_weight, all_classes=all_classes, forget_class=forget_class,\
                img_ext=img_ext)

    return net, dataloader, optim, z_random, identifier, sample_dir, checkpoint_dir, epoch_length, epochs,\
           num_steps,save_steps, collect_interval, log_interval, csv_file, device, grid_size



def train(model, folder, num_steps, batch_size, latent_dim=512, save_steps=None, collect_interval='epoch', log_interval=10,\
          kl_weight=1., uniformity_weight=0., all_classes=[0, 1], forget_class=1,\
          img_ext='jpg', classifier_path="../../data/CelebA/cnn/cnn_10.pth", data_path="../../data/CelebA/dataset", max_data=None):
   
    """
    Train a Variational Autoencoder (VAE) on the CelebA dataset.

    Parameters
    ----------
    model : str or torch.nn.Module
        Path to a saved model or a model instance.
    folder : str
        Directory to store training results.
    num_steps : int
        Total number of training steps.
    batch_size : int
        Number of samples per training batch.
    latent_dim : int, optional
        Dimensionality of the latent space. Defaults to 2.
    save_steps : int or None, optional
        Steps interval at which to save the model. Defaults to None.
    collect_interval : str, optional
        Interval for collecting samples, 'epoch' or 'step'. Defaults to 'epoch'.
    log_interval : int, optional
        Interval for logging training progress. Defaults to 10.
    kl_weight : float, optional
        Weight for the KL divergence loss. Defaults to 1.
    uniformity_weight : float, optional
        Weight for the uniformity loss. Defaults to 1e4.
    all_classes : list, optional
        List of class labels to use. Defaults to [0, 1].
    forget_class : int, optional
        Class label to forget during training. Defaults to 1.
    img_ext : str, optional
        Image file extension for saved samples. Defaults to 'jpg'.
    classifier_path : str, optional
        Path to a pre-trained classifier model. Defaults to "../../data/CelebA/cnn/cnn_10.pth".
    data_path : str, optional
        Path to the CelebA dataset. Defaults to "../../data/CelebA/dataset".

    Returns
    -------
    None
    """
    # ---------------------------------------------------
    # Setup
    # ---------------------------------------------------
    net, dataloader, optim, z_random, identifier, sample_dir, checkpoint_dir, epoch_length, epochs,\
    num_steps, save_steps, collect_interval, log_interval, csv_file, device, grid_size \
    = init(model, folder, num_steps, batch_size, latent_dim=latent_dim, save_steps=save_steps, collect_interval=collect_interval,\
           log_interval=log_interval, kl_weight=kl_weight, uniformity_weight=uniformity_weight, orthogonality_weight=0.,\
           all_classes=all_classes, forget_class=forget_class, img_ext=img_ext, classifier_path=classifier_path, data_path=data_path, max_data=max_data)
    process_batch = get_processor(net, identifier, z_random, (kl_weight, uniformity_weight), optim, all_classes, forget_class)    
    log_results = get_logger(net, identifier, dataloader, (kl_weight, uniformity_weight), csv_file, log_interval, all_classes)
    save = get_saver(net, save_steps, checkpoint_dir, epoch_length)
    collect_samples = get_collector(sample_dir, collect_interval, grid_size, img_ext)   

    # ---------------------------------------------------
    # Main training loop
    # ---------------------------------------------------
    global_step = 0
    for _ in tqdm(range(1, epochs + 1), desc="Epochs"):
        for _, (real_img, _) in enumerate(dataloader['original']):
            global_step += 1
            real_img = real_img.to(device)
            # -- Process a single batch
            rec_loss, kl_loss, unif_loss, generated_img, logits, elapsed_time = process_batch(real_img)
            loss = rec_loss + kl_weight * kl_loss + uniformity_weight * unif_loss
            log_results(step=global_step, losses=[rec_loss, kl_loss, unif_loss, loss], elapsed_time=elapsed_time, real_img=real_img, generated_img=generated_img, logits=logits)
            save(step=global_step)
            collect_samples(generated_img, step=global_step)
    viz.summarize_training(folder)
