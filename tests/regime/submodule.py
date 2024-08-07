"""
Contains a class that is used to test the RegimeMeta class and hyperparameter decorator.
"""

import torch

from regime.utils import RegimeMeta, hyperparameter


class ExampleClassB(RegimeMeta):
    """
    Example class B that uses the RegimeMeta metaclass.
    """

    @staticmethod
    def resource_name() -> str:
        return "example_b"

    @hyperparameter("beta")
    def __call__(
        self, input_data: torch.Tensor, beta: float, device: torch.device
    ) -> float:
        return input_data.sum().item() + beta
