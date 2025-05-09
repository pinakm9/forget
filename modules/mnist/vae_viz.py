import torch
import numpy as np
from vae import VAE
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os, sys, glob, json
import matplotlib.lines as mlines

sys.path.append(os.path.abspath('../modules'))
import utility as ut
import classifier
import pandas as pd
import vae 

def box_muller(steps):
    """
    Generate a grid of points in 2D space using the Box-Muller transform.

    Parameters
    ----------
    steps : int
        The number of points to generate in each dimension.

    Returns
    -------
    torch.Tensor
        A tensor of shape (steps**2, 2) containing the generated points.
    """
    p = []
    u1 = np.linspace(0.01, 0.99, steps)
    u2 = np.linspace(0.01, 0.99, steps)
    for i in range(len(u1)):
        for j in range(len(u2)):
            z1 = np.sqrt(-2*np.log(u1[i]))*np.cos(2*np.pi*u2[j])
            z2 = np.sqrt(-2*np.log(u1[i]))*np.sin(2*np.pi*u2[j])
            coor = [z1, z2]
            p.append(torch.tensor(coor, dtype = torch.float).unsqueeze(0))
    return torch.cat(p, 0)



def generate_random_samples(model, num_samples=25, latent_dim=2):
    """
    Generate a grid of random images using a trained VAE model.

    Parameters
    ----------
    model : VAE or str
        The VAE model to use for generating images. Can be a VAE instance 
    num_samples : int, optional
        The number of images to generate. Default is 25.
    latent_dim : int, optional
        The dimensionality of the latent space. Default is 2.

    Notes
    -----
    This function generates a grid of random images using the decoder of the
    VAE model. The images are sampled from a standard normal distribution in
    the latent space.
    """
    net = model
    net.eval()  # Set model to evaluation mode
    device = net.device
    
    # Sample random latent vectors from a standard normal distribution
    z_random = torch.randn(num_samples, latent_dim).to(device)

    # Decode the latent vectors into images
    with torch.no_grad():
        generated_images = net.decoder(z_random)  # Shape: (num_samples, 784)
    
    generated_array = generated_images.cpu().numpy().reshape(num_samples, 28, 28)

    # Define grid size for visualization
    grid_size = int(num_samples ** 0.5)  # Square grid if possible
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))

    for i in range(grid_size):
        for j in range(grid_size):
            img_idx = i * grid_size + j
            if img_idx < num_samples:
                axes[i, j].imshow(generated_array[img_idx], cmap="gray")
                axes[i, j].axis("off")

    plt.show()
    return generated_images


def compare_generated_samples(model_a, model_b, num_samples=25, latent_dim=2):
    """
    Generate a side-by-side grid of random images using two VAE models for the same latent vectors.

    Parameters
    ----------
    model_a : VAE
        The first VAE model.
    model_b : VAE
        The second VAE model.
    num_samples : int
        The number of images to generate.
    latent_dim : int
        The dimensionality of the latent space.
    """
    model_a.eval()
    model_b.eval()
    
    device = model_a.device
    z_random = box_muller(int(np.sqrt(num_samples)))
    z_random = torch.tensor(z_random, dtype=torch.float32).to(device) #torch.randn(num_samples, latent_dim).to(device)

    with torch.no_grad():
        images_a = model_a.decoder(z_random).cpu().numpy().reshape(num_samples, 28, 28)
        images_b = model_b.decoder(z_random).cpu().numpy().reshape(num_samples, 28, 28)

    grid_size = int(num_samples ** 0.5)
    fig, axes = plt.subplots(grid_size, grid_size * 2, figsize=(2 * grid_size, grid_size))

    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            if idx < num_samples:
                ax_a = axes[i, j * 2]
                ax_b = axes[i, j * 2 + 1]

                ax_a.imshow(images_a[idx], cmap='gray')
                ax_b.imshow(images_b[idx], cmap='gray')

                ax_a.axis('off')
                ax_b.axis('off')

    # Add centered titles above each grid
    fig.suptitle('', y=1.0)
    fig.text(0.25, 0.98, "Original model", ha='center', fontsize=16)
    fig.text(0.75, 0.98, "After unlearning", ha='center', fontsize=16)

    # Add vertical line between grids
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    # Add vertical line between the grids
    line = mlines.Line2D([0.5, 0.5], [0, 1], transform=fig.transFigure, color='black', linewidth=1, linestyle='-')
    fig.add_artist(line)

    plt.tight_layout()
    plt.show()
    return images_a, images_b










def evolve(folder, window=1):
    """
    Plot the evolution of training metrics across epochs.

    Parameters
    ----------
    folder : str
        The base folder containing the `checkpoints` directory with the training log.
    window : int, optional
        The rolling window size for the moving average of the loss and metrics. Default is 1.

    Notes
    -----
    This function plots the evolution of training metrics across epochs. The
    metrics are read from the `training_log.csv` file in the `checkpoints`
    directory. The plots are saved as `evolution.png` in the same folder.

    The plots include the total loss, orthogonality loss, uniformity loss, the
    fraction of forget digits, the margin, and the FID and IS of the generated
    images.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    data = pd.read_csv(f"{folder}/checkpoints/training_log.csv")
    config = ut.get_config(folder)
    # window = 1#int(max(1, config["training"]["epoch_length"]["value"] / config["experiment"]["log_interval"]["value"]))

    # plot Total Loss vs Step
    axes[0, 0].semilogy(data["Step"], data["Total Loss"].rolling(window).mean())
    axes[0, 0].set_ylabel("Total Loss")


    # plot Orthogonality Loss vs Step
    try:
        axes[0, 1].semilogy(data["Step"], data["Orthogonality Loss"].rolling(window).mean())
        axes[0, 1].set_ylabel("Orthogonality Loss")
    except: 
        pass
    
    # plot Uniformity Loss vs Step
    try:
        axes[0, 2].semilogy(data["Step"], data["Uniformity Loss"].rolling(window).mean())
        axes[0, 2].set_ylabel("Uniformity Loss")
    except: 
        pass
    
    try:
        # plot fraction of forget digit vs Step
        forget_digit = config['experiment']['forget_digit']["value"]
        axes[1, 0].plot(data["Step"], data[f"{forget_digit} Fraction"])
        axes[1, 0].set_ylabel(f"Fraction of {forget_digit}s")
        axes[1, 0].set_xlabel("Step")
    except:
        pass

    try:
        # plot Margin vs Step
        axes[1, 1].plot(data["Step"], data["Margin"])
        axes[1, 1].set_ylabel("Margin")
    except:
        pass

    try:
        # plot quality vs Step
        axes[1, 2].plot(data["Step"], data["FID"], label="FID")
        axes[1, 2].plot(data["Step"], data["IS"], label="IS")
        axes[1, 2].set_ylabel("Image Quality")
        axes[1, 2].legend()
    except:
        pass


    fig.supxlabel("Step")
    plt.savefig(f"{folder}/evolution.png", bbox_inches="tight")


@ut.timer
def summarize_training(folder, window=1, total_duration=15):
    """
    Summarize the training run by plotting the evolution of metrics and saving an animation of the generated samples.

    Parameters:
        folder (str): The base folder containing the `checkpoints` directory with the training log.
        window (int, optional): The rolling window size for the moving average of the loss and metrics. Default is 1.
        total_duration (int or None, optional): The total duration of the video in seconds. If None, no video is saved. Defaults to 15.

    Notes:
        This function assumes that the training log is in the `checkpoints` directory inside the provided folder.
        The plot is saved as `evolution.png` in the same folder, and the animation is saved as `sample_evolution.mp4` in the `samples` folder.
    """
    
    evolve(folder, window)
    config = ut.get_config(folder)
    if isinstance(total_duration, int):
        ut.stitch(f"{folder}/samples", config["experiment"]["img_ext"]["value"], f"{folder}/samples/sample_evolution.mp4", total_duration, delete_images=True)





# @ut.timer
# def evolve(model, folder, num_samples=25, fps=12, total_frames=100):
#     """
#     Creates an animation showing the evolution of generated grayscale images over different model checkpoints.

#     Parameters:
#         model: The generative model with a `.decoder` method.
#         folder (str): Base folder containing 'checkpoints' where model states are stored.
#         num_samples (int): Number of images to generate per frame.
#         fps (int): Frames per second for the final animation.
#     """
#     device = model.device
#     checkpoints_dir = os.path.join(folder, "checkpoints")
#     # Get sorted list of checkpoint files.
#     pth_files = sorted(glob.glob(os.path.join(checkpoints_dir, "*.pth")))
#     if len(pth_files) < total_frames:
#         total_frames = len(pth_files) 

#     # Ensure samples directory exists
#     samples_dir = os.path.join(folder, "samples")
#     os.makedirs(samples_dir, exist_ok=True)

#     # Determine grid size (square grid if possible)
#     grid_size = int(np.ceil(np.sqrt(num_samples)))

#     # Sample random latent vectors using Box-Muller
#     z_random = box_muller(grid_size).to(device)

#     # Create figure and axes for animation
#     fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
#     if grid_size == 1:
#         axes = np.array([axes])
#     else:
#         axes = axes.flatten()

#     # Initialize the image plots with blank grayscale images
#     image_plots = [ax.imshow(np.zeros((28, 28)), cmap="gray", vmin=0, vmax=1, animated=True) for ax in axes]
#     for ax in axes:
#         ax.axis("off")

#     # Add a title
#     title = fig.suptitle("Fraction of 1s = ?")

#     def update(frame_idx):
#         """
#         Function to update the animation with new generated grayscale images.
#         """
#         # Load model checkpoint
#         model.load_state_dict(torch.load(pth_files[frame_idx], map_location=device))
        
#         # Generate images
#         with torch.no_grad():
#             generated_images = model.decoder(z_random)
        
#         # Count how many generated digits are classified as '1'
#         num_ones = classifier.count_digit(generated_images, 1, device=device)

#         # Convert to numpy and reshape to (num_samples, 28, 28)
#         generated_images = generated_images.cpu().numpy().reshape(num_samples, 28, 28)

#         # Update each subplot with the new grayscale image
#         for i, img_plot in enumerate(image_plots):
#             if i < num_samples:
#                 img_plot.set_array(generated_images[i])

#         # Update title with fraction of 1s
#         title.set_text(f"Fraction of 1s = {num_ones / num_samples:.3f}")
#         return image_plots + [title]

#     # Create the animation
#     ani = animation.FuncAnimation(fig, update, frames=np.linspace(0, len(pth_files)-1, total_frames, dtype=int), interval=1000//fps, blit=False)

#     # Save as MP4
#     output_video = os.path.join(samples_dir, "evolution.mp4")
#     ani.save(output_video, writer="ffmpeg", fps=fps)

#     print(f"Animation saved as {output_video}")
