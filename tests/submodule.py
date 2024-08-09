"""
Contains a class that is used to test the RegimeMeta class and hyperparameter decorator.
"""

import torch

from regime import Node, hyperparameter


class ExampleClassB(Node):  # pylint: disable=too-few-public-methods
    """
    Example class B that uses the RegimeMeta metaclass.
    """

    @hyperparameter("beta")
    def __call__(
        self, input_data: torch.Tensor, beta: float, device: torch.device
    ) -> float:
        return input_data.sum().item() + beta
