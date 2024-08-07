"""
Utilities for working with Regime objects.
"""

import inspect
from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Dict


def hyperparameter(*args):
    """
    The hyperparameter decorator for Regime compatible classes. Allows for the explicit tagging of
    hyperparameters in a class, so easier configuration management can occur.

    Args:
        *args: The hyperparameters to tag.

    Returns:
        The decorator.
    """

    def decorator(func):
        """
        The decorator.

        Args:
            func: The function to decorate.

        Returns:
            The decorated function.
        """
        if not hasattr(func, "hyperparameters"):
            func.hyperparameters = []
        func.hyperparameters.extend(args)
        return func

    return decorator


class HyperparameterMeta(type(ABC)):  # was originally just: type
    """
    Metaclass for RegimeMeta. Automatically collects hyperparameters from the class (assuming it
    has been decorated with the hyperparameter decorator) and stores them in a dictionary.
    """

    def __new__(mcs, name, bases, dct):
        dct["_hyperparameters"] = {}
        for _, attr in dct.items():  # attr_name, attr
            if callable(attr) and hasattr(attr, "hyperparameters"):
                for param in attr.hyperparameters:
                    dct["_hyperparameters"][param] = None
        return super().__new__(mcs, name, bases, dct)


class RegimeMeta(metaclass=HyperparameterMeta):
    """
    The RegimeMeta class. This class is used to define the structure of a Regime compatible object.
    It can automatically collect hyperparameters from the class and store them in a dictionary, as
    well as ensures the class has a resource_name method and provides a default implementation for
    the edges method(s) (used to direct the data flow of the algorithm).
    """

    @staticmethod
    @abstractmethod
    def resource_name() -> str:
        """
        The name of the resource that the algorithm produces.

        Returns:
            A string representing the resource name.
        """
        raise NotImplementedError("resource_name must be implemented in the subclass.")

    @classmethod
    def edges(cls) -> List[Tuple[Any, Any, int]]:
        """
        Edges that bring data into the algorithm and distribute the output of the algorithm
        to the next algorithm in the pipeline.

        Used for constructing Regime objects.

        Returns:
            A list of 3-tuples, where the first element is the source, the second element is the
            target, and the third element is the order of the argument in the
            target's __call__ method.
        """
        return cls.source_edges() + cls.target_edges()

    # for code reuse, the following methods are broken down into smaller methods
    # often, in more complicated Regimes, the source_edges and target_edges methods
    # will be overridden to provide more specific behavior (e.g., different training data)
    # or their target destinations will have to be overridden to provide more specific behavior

    @classmethod
    def source_edges(cls) -> List[Tuple[Any, Any, int]]:
        """
        Edges that bring data into the algorithm.

        Returns:
            A list of 3-tuples, where the first element is the source, the second element is the
            target, and the third element is the order of the argument in the
            target's __call__ method.
        """
        return cls.arg_only_edges() + cls.hyperparameter_edges()

    @classmethod
    def arg_only_edges(cls) -> List[Tuple[Any, Any, int]]:
        """
        Edges that bring arguments except hyperparameters into the algorithm.

        Returns:
            A list of 3-tuples, where the first element is the source, the second element is the
            target, and the third element is the order of the argument in the
            target's __call__ method.
        """
        parameters_mapping_proxy = inspect.signature(cls.__call__).parameters
        return [
            (key, cls, arg_order)
            for arg_order, key in enumerate(parameters_mapping_proxy.keys())
            if key != "self" and key not in collect_hyperparameters(cls)
        ]

    @classmethod
    def hyperparameter_edges(cls) -> List[Tuple[Any, Any, int]]:
        """
        Edges that bring hyperparameters into the algorithm.

        Returns:
            A list of 3-tuples, where the first element is the source, the second element is the
            target, and the third element is the order of the argument in the
            target's __call__ method.
        """
        return [
            (key, cls, arg_order)
            for arg_order, key in enumerate(collect_hyperparameters(cls).keys())
        ]

    @classmethod
    def target_edges(cls) -> List[Tuple[Any, Any, int]]:
        """
        Edges that distribute the output of the algorithm to the next algorithm in the pipeline.

        Returns:
            A list of 3-tuples, where the first element is the source, the second element is the
            target, and the third element is the order of the argument for the target.
        """
        return [(cls, cls.resource_name(), 0)]


def collect_hyperparameters(cls: Any) -> Dict[str, None]:
    """
    Collect hyperparameters from a class.

    Args:
        cls: The class to collect hyperparameters from.

    Returns:
        A dictionary of hyperparameters. The keys are the hyperparameter names and the values are
        their corresponding (default?) values.
    """
    return cls._hyperparameters  # pylint: disable=protected-access


def module_path_to_dict(module_path, callback=None) -> dict:
    """
    Convert a module path to a nested dictionary.

    Args:
        module_path: The module path.
        callback: The callback function; appends to the dictionary at the most bottom level.

    Returns:
        The nested dictionary.
    """
    parts = module_path.split(".")
    nested_dict = current = {}
    for part in parts:
        current[part] = {}
        current = current[part]
    if callback is not None:
        callback(current)
    return nested_dict


def make_hyperparameters_dict(cls, include_hyperparameters=True) -> dict:
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
        hyperparameters: Dict[str, None] = collect_hyperparameters(cls)

    if len(hyperparameters) == 0:
        return {}  # no hyperparameters to include
    return module_path_to_dict(
        cls.__module__,
        callback=lambda current: current.update({cls.__name__: hyperparameters}),
    )


def merge_dicts(existing, new):
    """
    Merge two dictionaries.

    Args:
        existing: The first dictionary.
        new: The second dictionary.

    Returns:
        The merged dictionary.
    """
    for key, value in new.items():
        if isinstance(value, dict):
            existing[key] = merge_dicts(existing.get(key, {}), value)
        else:
            existing[key] = value
    return existing
