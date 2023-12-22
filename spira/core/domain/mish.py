import torch
from torch.nn.functional import softplus


# adapted from https://github.com/digantamisra98/Mish/blob/master/Mish/Torch/mish.py
class Mish(torch.nn.Module):
    """
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Examples:
        >>> m = Mish()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def __init__(self):
        """
        Init method.
        """
        super().__init__()

    @staticmethod
    def forward(inp):
        """
        Forward pass of the function.
        """
        return inp * torch.tanh(softplus(inp))
