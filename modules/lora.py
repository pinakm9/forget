import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, alpha=1.0, bias=True):
        """
        LoRA version of a linear layer.
        
        Parameters
        ----------
        in_features : int
            Size of each input sample.
        out_features : int
            Size of each output sample.
        r : int, optional
            Rank of the low-rank decomposition. Default is 4.
        alpha : float, optional
            Scaling factor for the LoRA updates. Default is 1.0.
        bias : bool, optional
            If set to False, the layer will not learn an additive bias. Default is True.
        """
        super(LoRALinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 1.0

        # Base weight is created as a parameter, but it will be frozen.
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.weight.requires_grad = False  # frozen base weight

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

        # Initialize the low-rank adaptation parameters.
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(out_features, r) * 0.01)
    

    def forward(self, x):
        """
        Computes the output of the LoRA linear layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """

        # Effective weight is the sum of the frozen base weight and the low-rank update.
        return F.linear(x, self.weight + self.scaling * torch.mm(self.lora_B, self.lora_A), self.bias)




def convert_linear_to_lora(linear_layer: nn.Linear, lora_r: int, lora_alpha: float) -> LoRALinear:
    """
    Convert a PyTorch nn.Linear layer to a LoRA layer.

    The given linear layer is copied and frozen, and a LoRALinear instance is created
    with the same dimensions. The LoRALinear instance is then returned.

    Parameters
    ----------
    linear_layer : nn.Linear
        The PyTorch nn.Linear layer to be converted.
    lora_r : int
        The rank of the low-rank decomposition.
    lora_alpha : float
        The scaling factor for the LoRA updates.

    Returns
    -------
    LoRALinear
        The converted LoRALinear layer.
    """
    in_features = linear_layer.in_features
    out_features = linear_layer.out_features
    bias = linear_layer.bias is not None

    # Create a LoRALinear instance with the same dimensions.
    lora_layer = LoRALinear(in_features, out_features, r=lora_r, alpha=lora_alpha, bias=bias)
    
    # Copy the pretrained weight and freeze it.
    with torch.no_grad():
        lora_layer.weight.copy_(linear_layer.weight)
    lora_layer.weight.requires_grad = False

    # If there is a bias, copy and freeze it.
    if bias:
        with torch.no_grad():
            lora_layer.bias.copy_(linear_layer.bias)
        lora_layer.bias.requires_grad = False

    return lora_layer

