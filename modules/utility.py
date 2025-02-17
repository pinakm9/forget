# A helper module for various sub-tasks
from time import time
import torch
import torch.nn as nn
import glob, os, json
import numpy as np



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
