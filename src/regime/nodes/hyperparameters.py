"""
All functions related managing hyperparameters are defined here (except for HyperparameterMeta).
"""

from typing import Dict

from regime import Node
from regime.utils import module_path_to_dict


def make_hyperparameters_dict(cls: Node, include_hyperparameters=True) -> dict:
    """
    Generate a nested dictionary of hyperparameters for the given class (including class name).
    Whether to include the hyperparameters themselves is optional, but the class name is always
    included; default is to include hyperparameters.

    Args:
        cls: The Python class to generate the hyperparameters dictionary for.
        include_hyperparameters: Whether to include the hyperparameters themselves in
        the dictionary.

    Returns:
        A nested dictionary of the class name and its hyperparameters.
    """
    hyperparameters: Dict = {}
    if include_hyperparameters:
        hyperparameters: Dict[str, None] = cls.hyperparameters()

    if len(hyperparameters) == 0:
        return {}  # no hyperparameters to include
    return module_path_to_dict(
        cls.__module__,
        callback=lambda current: current.update({cls.__name__: hyperparameters}),
    )
