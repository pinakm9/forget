# A helper module for various sub-tasks
from time import time
import torch
import torch.nn as nn
import glob, os, json
import numpy as np
import platform
import cv2, re



def timer(func):
	"""
	Timing wrapper for a generic function.
	Prints the time spent inside the function to the output.
	"""
	def new_func(*args, **kwargs):
		start = time()
		val = func(*args,**kwargs)
		end = time()
		print(f'Time taken by {func.__name__} is {end-start:.4f} seconds')
		return val
	return new_func



class ExperimentLogger:
    def __init__(self, folder, description):
        """
        Constructor for ExperimentLogger.
        
        Parameters
        ----------
        folder : str
            The folder where log files are saved.
        description : str
            A description of the experiment.
        
        Attributes
        ----------
        folder : str
            The folder where log files are saved.
        description : str
            A description of the experiment.
        objects : list
            A list of objects which are to be logged.
        logFile : str
            The path to the log file.
        """
        self.folder = folder
        self.description = description
        self.objects = []
        self.logFile = folder + '/experiment_log.txt'
        with open(self.logFile, 'w') as file:
            file.write("=======================================================================\n")
            file.write("This is a short description of this experiment\n")
            file.write("=======================================================================\n")
            descriptionWithNewLine= description.replace('. ', '.\n')
            file.write(f"{descriptionWithNewLine}\n\n\n")
        
    def addSingle(self, object):
        """
        Logs a single object's attributes to the log file.
        
        Parameters
        ----------
        object : object
            The object to be logged.
        """
        self.objects.append(object)
        with open(self.logFile, 'a') as file:
            file.write("=======================================================================\n")
            file.write(f"Object: {object.name}\n")
            file.write("=======================================================================\n")
            for key, value in object.__dict__.items():
                file.write(f"{key} = {value}\n")
                file.write("-----------------------------------------------------------------------\n")
            file.write("\n\n")
    
    def add(self, *objects):
        """
        Logs multiple objects' attributes to the log file.
        
        Parameters
        ----------
        objects : object
            The objects to be logged.
        """
        for object in objects:
            self.addSingle(object)


def makedirs(*args):
    """
    Creates a directory if it doesn't already exist.

    Parameters
    ----------
    *args : str
        The paths to the directories to be created.
    """
    for arg in args:
        if not os.path.exists(arg):
            os.makedirs(arg)


def count_params(model):
    """
    Counts the total number of parameters in a model, and the number of trainable parameters.

    Parameters
    ----------
    model : torch.nn.Module
        The model for which to count the parameters.

    Returns
    -------
    total_params : int
        The total number of parameters in the model.
    trainable_params : int
        The number of trainable parameters in the model.
    """
    total_params = sum(p.numel() for p in model.parameters())  # All parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # Only trainable

    return total_params, trainable_params


def flatten_params(model):
    """
    Gathers all parameters from `model` into one 1-D Tensor.
    Returns:
      flattened_params (torch.Tensor): Flattened parameters (no grad).
      shapes (list of torch.Size): List of original parameter shapes.
    """
    param_list = []
    shapes = []
    for p in model.parameters():
        shapes.append(p.shape)
        param_list.append(p.data.view(-1))  # Flatten each parameter to 1D
    flattened_params = torch.cat(param_list)
    return flattened_params, shapes

def unflatten_params(flattened_params, shapes):
    """
    Slices the 1-D `flattened_params` back into separate tensors
    whose shapes are in `shapes`.
    Returns:
      param_list (list of torch.Tensor): List of parameter tensors
      matching the original shapes.
    """
    param_list = []
    idx = 0
    for shape in shapes:
        size = torch.prod(torch.tensor(shape))
        param_list.append(flattened_params[idx: idx + size].view(shape))
        idx += size
    return param_list

def set_model_params(model, param_tensors):
    """
    Copies data from each tensor in `param_tensors` into the 
    corresponding parameter of `model`.
    """
    for p, new_p in zip(model.parameters(), param_tensors):
        p.data.copy_(new_p)  # or p.data = new_p.data





def freeze_all(model: nn.Module) -> nn.Module:
    """
    Freeze all parameters in the given model 
    
    Args:
        model (nn.Module): The model whose layers you want to modify.
    
    Returns:
        nn.Module: The same model with modified requires_grad attributes.
    """
    for param in model.parameters():
        param.requires_grad = False

    return model



def freeze_all_but_last(model: nn.Module) -> nn.Module:
    """
    Freeze all parameters in the given model except those in the last
    child module. The "last" layer is determined by the final child
    in model.children().
    
    Args:
        model (nn.Module): The model whose layers you want to modify.
    
    Returns:
        nn.Module: The same model with modified requires_grad attributes.
    """
    # 1) Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # 2) Unfreeze the parameters in the last child module
    last_child = list(model.children())[-1]
    for param in last_child.parameters():
        param.requires_grad = True

    return model


def freeze_all_but_first(model: nn.Module) -> nn.Module:
    """
    Freeze all parameters in the given model except those
    in the first child module.
    """
    # 1) Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # 2) Unfreeze the parameters in the first child module
    first_child = list(model.children())[0]
    for param in first_child.parameters():
        param.requires_grad = True

    return model


def get_trainable_params(model: nn.Module):
    """
    Return a list of all parameters in the model that require gradients.
    
    Args:
        model (nn.Module): The PyTorch model.
    
    Returns:
        List[torch.Tensor]: A list of parameters that will be updated during backpropagation.
    """
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    return trainable_params#torch.cat([p.view(-1) for p in trainable_params], dim=0).to(model.device)


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif platform.system() == "Darwin":
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device



def stitch(image_folder, img_ext='jpg', output_video=None, total_duration=30, fps=None, delete_images=False, max_images=None):
    """
    Stitch images in the specified folder into a video.

    Parameters:
        image_folder (str): The folder containing the images to be stitched.
        img_ext (str, optional): The image file extension. Defaults to 'jpg'.
        output_video (str, optional): The path to the output video file. If None, the video will be saved to `video.mp4` inside the image folder. Defaults to None.
        total_duration (int, optional): The total duration of the video in seconds. Defaults to 30.
        fps (int, optional): The frames per second of the video. If None, the fps will be determined based on the total duration. Defaults to None.
        delete_images (bool, optional): Whether to delete the images after stitching. Defaults to False.
        max_images (int, optional): The maximum number of images to include in the video. If None, all images will be included. Defaults to None.

    Raises:
        ValueError: If no images are found in the specified folder.

    Notes:
        The images are sorted in natural order based on their filenames.
        The first image is read to get the width and height of the video.
    """
    if output_video is None:
        output_video = os.path.join(image_folder, "video.mp4")
    # Get all image files in sorted order
    def natural_key(path):
        """
        Split the filename into alternating text and number chunks,
        so that 'sample_10.jpg' → ['sample_', 10, '.jpg'].
        """
        filename = os.path.basename(path)
        parts = re.split(r'(\d+)', filename)
        return [int(p) if p.isdigit() else p.lower() for p in parts]

    # …

    image_files = glob.glob(os.path.join(image_folder, f"*.{img_ext}"))
    image_files = sorted(image_files, key=natural_key)

    if not image_files:
        raise ValueError("No images found in the specified folder.")

    # Read the first image to get width and height
    first_frame = cv2.imread(image_files[0])
    height, width, _ = first_frame.shape

    # Determine FPS based on total duration
    if fps is None:
        fps = max(1, len(image_files) / total_duration)  # Avoid zero division

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for img_path in image_files if max_images is None else image_files[:max_images]:
        frame = cv2.imread(img_path)
        out.write(frame)

    out.release()
    # Delete images if flag is set
    if delete_images:
        for img_path in image_files:
            os.remove(img_path)
    print(f"Video saved at {output_video}")

# Example usage:
# images_to_video("path_to_images", "output_video.mp4", total_duration=5)

def get_config(folder):
    with open(f"{folder}/config.json", 'r') as f:
            config = json.load(f)
    return config


def torch_cov(x):
    """
    Compute the covariance matrix for x, where x is a 2D tensor of shape (N, D).
    """
    n = x.size(0)
    x_mean = torch.mean(x, dim=0, keepdim=True)
    x_centered = x - x_mean
    cov = x_centered.t() @ x_centered / (n - 1)
    return cov


def get_file_count(folder_path):
    """
    Count the number of files in a given folder.

    Parameters
    ----------
    folder_path : str
        The path to the folder to count files in.

    Returns
    -------
    int
        The number of files in the folder.
    """
    return len([
        name for name in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, name))
    ])