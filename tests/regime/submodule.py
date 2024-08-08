"""
Contains a class that is used to test the RegimeMeta class and hyperparameter decorator.
"""

import torch

from regime import Node, hyperparameter


class ExampleClassB(Node):
    """
    Example class B that uses the RegimeMeta metaclass.
    """

    def __init__(self, resource_name: str):
        super().__init__(resource_name)

    @hyperparameter("beta")
    def __call__(
        self, input_data: torch.Tensor, beta: float, device: torch.device
    ) -> float:
        return input_data.sum().item() + beta
