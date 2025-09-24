import matplotlib.pyplot as plt 
import pandas as pd
import utility as ut
import numpy as np
from torchvision.utils import make_grid
import math

def evolve(folder, window=1):
    """
    Plots the evolution of Total Loss, Orthogonality Loss, Uniformity Loss, Fraction of forget class, Margin, and Image Quality metrics over the training steps.

    Parameters:
        folder (str): The base folder containing the `checkpoints` directory with the training log.
        window (int, optional): The rolling window size for the moving average of the loss and metrics. Default is 1.

    Notes:
        This function assumes that the training log is in the `checkpoints` directory inside the provided folder.
        The plot is saved as `evolution.png` in the same folder.
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
        # plot fraction of forget class vs Step
        forget_class = config['experiment']['forget_class']["value"]
        axes[1, 0].plot(data["Step"], data[f"{forget_class} Fraction"])
        axes[1, 0].set_ylabel(f"Fraction of {forget_class}s")
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
        try:
            ut.stitch(f"{folder}/samples", config["experiment"]["img_ext"]["value"], f"{folder}/samples/sample_evolution.mp4", total_duration, delete_images=True)
        except:
            pass
    plt.close("all")





def show(imgs, nrow=4, filename=None):
    """
    Display a batch of images with their indices as titles.
    
    Args:
        imgs: tensor [B,3,H,W] in [0,1]
        nrow: number of images per row
        filename: if given (str), save the figure instead of just showing
    """
    B = imgs.shape[0]
    ncol = nrow
    nrows = (B + ncol - 1) // ncol  # ceil division

    fig, axes = plt.subplots(nrows, ncol, figsize=(ncol * 3, nrows * 3))

    # Ensure axes is always iterable
    if nrows == 1 and ncol == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < B:
            img = imgs[i].permute(1, 2, 0).cpu().numpy()
            ax.imshow(img)
            ax.set_title(f"{i}", fontsize=10)
            ax.axis("off")
        else:
            ax.axis("off")

    plt.tight_layout()

    if filename:
        plt.savefig(filename, bbox_inches="tight", dpi=300)
        plt.close(fig)
    else:
        plt.show()





def plot_transformation(list_images_a, list_images_b, list_index, save_path=None):
    images_a = np.concatenate([list_images_a[i][j] for i, j in enumerate(list_index)], axis=0)
    images_b = np.concatenate([list_images_b[i][j] for i, j in enumerate(list_index)], axis=0)
    
    num_samples = images_a.shape[0]
    
    images_a = images_a.transpose(0, 2, 3, 1)
    images_b = images_b.transpose(0, 2, 3, 1)

    # images_a = (images_a - images_a.min()) / (images_a.max() - images_a.min())
    # images_b = (images_b - images_b.min()) / (images_b.max() - images_b.min())

    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2 + 1, 4))

    for idx in range(num_samples):
        axes[0, idx].imshow(images_a[idx])
        axes[0, idx].axis("off")

        axes[1, idx].imshow(images_b[idx])
        axes[1, idx].axis("off")


    # Add vertical labels to the left using fig.text
    fig.subplots_adjust(left=0.005, wspace=0.0, hspace=0.02)
    fig.text(0.0001, 0.7, "Original model", va='center', ha='left', rotation=90, fontsize=14)
    fig.text(0.0001, 0.3, "After unlearning", va='center', ha='left', rotation=90, fontsize=14)

    # Top labels for class
    fig.text(0.23, 0.9, "Golden Retriever", ha='center', va='bottom', fontsize=14)
    fig.text(0.67, 0.9, "Miscellaneous", ha='center', va='bottom', fontsize=14)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    return fig, axes





def compare_model_outputs_2(list_images_a, list_images_b, list_index, save_path=None):
    images_a = np.concatenate([list_images_a[i][j] for i, j in enumerate(list_index)], axis=0)
    images_b = np.concatenate([list_images_b[i][j] for i, j in enumerate(list_index)], axis=0)
    num_samples = images_a.shape[0]
    

    # Convert and normalize images
    images_a = images_a.transpose(0, 2, 3, 1)
    images_b = images_b.transpose(0, 2, 3, 1)

    # images_a = (images_a - images_a.min()) / (images_a.max() - images_a.min())
    # images_b = (images_b - images_b.min()) / (images_b.max() - images_b.min())

    # Combine A and B into a single list alternating
    images = []
    titles = []
    for i in range(num_samples):
        images.append(images_a[i])
        titles.append(f"Before-{i+1}")
        images.append(images_b[i])
        titles.append(f"After-{i+1}")

    total_images = len(images)  # 2 * num_samples
    grid_size = math.ceil(math.sqrt(total_images))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2))
    axes = axes.flatten()

    for i in range(len(images)):
        axes[i].imshow(images[i])
        axes[i].set_title(titles[i], fontsize=18)
        axes[i].axis("off")

    # Hide any remaining unused axes
    for i in range(len(images), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

    return fig, axes



