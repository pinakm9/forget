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

from pathlib import Path 
from functools import lru_cache 
import asyncio 
from typing import Iterable, Optional, Tuple, Union

import ipywidgets as widgets
import IPython.display as display 
from PIL import Image 

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
    u1 = np.linspace(0.05, .95, steps)
    u2 = np.linspace(0.05, .95, steps)
    for i in range(len(u1)):
        for j in range(len(u2)):
            z1 = np.sqrt(-2*np.log(u1[i]))*np.cos(2*np.pi*u2[j])
            z2 = np.sqrt(-2*np.log(u1[i]))*np.sin(2*np.pi*u2[j])
            coor = [z1, z2]
            p.append(torch.tensor(coor, dtype = torch.float).unsqueeze(0))
    return torch.cat(p, 0)



def generate_random_samples(model, num_samples=25, latent_dim=2, bm=False, title=None, seed=None):
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
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    net = model
    net.eval()  # Set model to evaluation mode
    device = next(net.parameters()).device
    
    # Sample random latent vectors from a standard normal distribution
    if bm:
        z_random = box_muller(int(np.sqrt(num_samples)))
        z_random = torch.tensor(z_random, dtype=torch.float32).to(device)
    else:
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

    if title is not None:
        fig.suptitle(title, fontsize=16)
    plt.show()
    return fig, axes



def compare_generated_samples(model_a, model_b, num_samples=25, latent_dim=2, bm=False, seed=None):
    """
    Compare VAE-generated samples using the same latent vectors in two separate figures.

    Args:
        model_a: First VAE model.
        model_b: Second VAE model.
        num_samples: Number of samples to generate.
        latent_dim: Dimensionality of latent space.
        bm: Use Box-Muller sampling instead of torch.randn.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of matplotlib figures: (fig_a, fig_b)
    """
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    model_a.eval()
    model_b.eval()

    device = next(model_a.parameters()).device
    if bm:
        z_random = box_muller(num_samples).to(device)
    else:
        z_random = torch.randn(num_samples, latent_dim).to(device)

    with torch.no_grad():
        images_a = model_a.decoder(z_random).cpu().view(-1, 28, 28).numpy()
        images_b = model_b.decoder(z_random).cpu().view(-1, 28, 28).numpy()

    grid_size = int(np.ceil(np.sqrt(num_samples)))

    def create_figure(images, title):
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size, grid_size))
        axes = axes.flatten()
        for idx in range(num_samples):
            axes[idx].imshow(images[idx], cmap='gray')
            axes[idx].axis('off')
        for idx in range(num_samples, len(axes)):
            axes[idx].axis('off')  # Hide extra axes if any
        # fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        return fig

    fig_a = create_figure(images_a, title="Original model")
    fig_b = create_figure(images_b, title="After unlearning")

    return fig_a, fig_b




def plot(images, title, saveas=False):
        num_samples = len(images)
        grid_size = int(np.sqrt(num_samples))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(4, 4))
        axes = axes.flatten()
        for idx in range(num_samples):
            axes[idx].imshow(images[idx].squeeze(), cmap='gray')
            axes[idx].axis('off')
        for idx in range(num_samples, len(axes)):
            axes[idx].axis('off')  # Hide extra axes if any
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        if not saveas:
            plt.show()
        else:
            plt.savefig(saveas, bbox_inches="tight", dpi=300)
        plt.close(fig)








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



def image_sequence_viewer(
    img_dir: Union[str, Path],
    *,
    method: str = "play",            # "play" or "async"
    interval_ms: int = 120,          # frame delay for play/async
    loop: bool = True,               # loop when reaching the end (async mode)
    max_side: int = 1000,            # downscale longest side for speed
    start_index: Optional[int] = None,  # integer filename to start at (e.g., 0)
):
    """
    Build a viewer for integer-indexed images in a folder (e.g., 0.png, 1.jpg, 2.png, ...).

    Parameters
    ----------
    img_dir : str | Path
        Folder containing images with integer stems.
    method : "play" | "async"
        "play" uses widgets.Play (simplest). "async" uses a Start/Stop button with speed & loop controls.
    interval_ms : int
        Frame delay in milliseconds.
    loop : bool
        Whether to loop when reaching the last frame (async method only).
    max_side : int
        If an image is larger than this on its longest side, it is downscaled for responsiveness.
    start_index : int | None
        If provided and present among filenames, the viewer starts at that index.

    Returns
    -------
    ipywidgets.VBox
        A container you can display() to show the UI.
    """
    img_dir = Path(img_dir)
    allowed = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
    idx2path = {}

    for p in img_dir.iterdir():
        if p.suffix.lower() in allowed:
            try:
                idx2path[int(str(p).split('/')[-1].split('.')[0].split('_')[-1])] = p
            except ValueError:
                pass

    if not idx2path:
        raise FileNotFoundError(
            f"No integer-indexed images found in {img_dir!s}. "
            f"Expected files like '0.png', '1.jpg', ..."
        )

    indices = sorted(idx2path)
    # slider runs over positions [0 .. len(indices)-1], so gaps in indices are fine
    def start_pos_from_index(sidx: Optional[int]) -> int:
        if sidx is None:
            return 0
        try:
            return indices.index(sidx)
        except ValueError:
            return 0

    pos = widgets.IntSlider(
        value=start_pos_from_index(start_index),
        min=0,
        max=len(indices) - 1,
        step=1,
        description="frame",
        continuous_update=False,
    )
    info = widgets.HTML()
    out = widgets.Output()

    @lru_cache(maxsize=256)
    def load(i: int, max_side_local: int = max_side):
        im = Image.open(idx2path[i]).convert("RGB")
        w, h = im.size
        m = max(w, h)
        if m > max_side_local:
            s = max_side_local / m
            im = im.resize((int(w * s), int(h * s)))
        return im

    def render(_=None):
        i = indices[pos.value]
        info.value = f"<b>Index:</b> {i} &nbsp;&nbsp; <b>File:</b> {idx2path[i].name}"
        with out:
            out.clear_output(wait=True)
            plt.figure()
            plt.imshow(load(i))
            plt.axis("off")
            plt.show()

    pos.observe(render, names="value")
    render()  # initial

    if method == "play":
        play = widgets.Play(
            interval=interval_ms, value=pos.value, min=pos.min, max=pos.max, step=1
        )
        widgets.jslink((play, "value"), (pos, "value"))
        controls = widgets.HBox([play, pos])
        ui = widgets.VBox([controls, info, out])
        return ui

    elif method == "async":
        ms = widgets.IntSlider(
            value=int(interval_ms), min=20, max=2000, step=10, description="ms/frame"
        )
        loop_chk = widgets.Checkbox(value=bool(loop), description="loop")
        start_btn = widgets.Button(description="▶ Start", button_style="primary", icon="play")
        stop_btn = widgets.Button(description="⏸ Stop", button_style="warning", icon="pause", disabled=True)

        anim_task = {"task": None}

        async def animate():
            start_btn.disabled = True
            stop_btn.disabled = False
            try:
                while True:
                    if pos.value >= pos.max:
                        if loop_chk.value:
                            pos.value = pos.min
                        else:
                            break
                    else:
                        pos.value += 1
                    await asyncio.sleep(ms.value / 1000.0)
            finally:
                start_btn.disabled = False
                stop_btn.disabled = True
                anim_task["task"] = None

        def on_start(_):
            if anim_task["task"] is None:
                anim_task["task"] = asyncio.create_task(animate())

        def on_stop(_):
            t = anim_task["task"]
            if t is not None:
                t.cancel()

        start_btn.on_click(on_start)
        stop_btn.on_click(on_stop)

        controls = widgets.HBox([start_btn, stop_btn, ms, loop_chk, pos])
        ui = widgets.VBox([controls, info, out])
        return ui

    else:
        raise ValueError("method must be 'play' or 'async'")
    




def compare_distributions(before, after, total=None, labels=None, fname=None):
    """
    Compare two categorical distributions with side-by-side bar plots.

    Args:
        before, after : 1D torch.Tensor or array-like
        total         : if None, normalize by sum of each vector;
                        if scalar, divide both by same total
        labels        : optional category labels for x-axis
        fname         : optional path to save figure
    """
    # convert to numpy
    b = before.detach().cpu().numpy() if isinstance(before, torch.Tensor) else np.asarray(before)
    a = after.detach().cpu().numpy()  if isinstance(after,  torch.Tensor) else np.asarray(after)

    # normalize
    if total is None:
        pb, pa = b / b.sum(), a / a.sum()
    else:
        pb, pa = b / total, a / total

    # colors based on comparison
    colors = ["green" if v1 > v0 else "red" if v1 < v0 else "blue"
              for v0, v1 in zip(pb, pa)]

    # plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    ax[0].bar(range(len(pb)), pb, edgecolor="black")
    ax[0].set_title("Original model", fontsize=14)
    ax[1].bar(range(len(pa)), pa, edgecolor="black")
    ax[1].set_title("After unlearning", fontsize=14)

    if labels is not None:
        for a in ax: a.set_xticks(range(len(labels)), labels)

    fig.subplots_adjust(wspace=0.1)
    if fname: fig.savefig(fname, bbox_inches="tight")
    return fig, ax





