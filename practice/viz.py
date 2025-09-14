import matplotlib.pyplot as plt 
import pandas as pd
import utility as ut

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