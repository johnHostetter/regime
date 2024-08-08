import inspect
from abc import abstractmethod
from typing import List, Tuple, Any

from regime.nodes.meta import HyperparameterMeta
from regime.nodes.hyperparameters import collect_hyperparameters


class Node(metaclass=HyperparameterMeta):
    """
    The Node class defines the structure of a Regime compatible object. It can automatically
    collect hyperparameters from the class and store them in a dictionary, as well as ensures the
    class has a resource_name method and provides a default implementation for the edges method(s)
    (used to direct the data flow of the algorithm).
    """

    def __init__(self, resource_name: str):
        self.resource_name = resource_name

    @staticmethod
    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("__call__ must be implemented in the subclass.")

    def edges(self) -> List[Tuple[Any, Any, int]]:
        """
        Edges that bring data into the algorithm and distribute the output of the algorithm
        to the next algorithm in the pipeline.

        Used for constructing Regime objects.

        Returns:
            A list of 3-tuples, where the first element is the source, the second element is the
            target, and the third element is the order of the argument in the
            target's __call__ method.
        """
        return self.source_edges() + self.target_edges()

    # for code reuse, the following methods are broken down into smaller methods
    # often, in more complicated Regimes, the source_edges and target_edges methods
    # will be overridden to provide more specific behavior (e.g., different training data)
    # or their target destinations will have to be overridden to provide more specific behavior

    def source_edges(self) -> List[Tuple[Any, Any, int]]:
        """
        Edges that bring data into the algorithm.

        Returns:
            A list of 3-tuples, where the first element is the source, the second element is the
            target, and the third element is the order of the argument in the
            target's __call__ method.
        """
        return self.arg_only_edges() + self.hyperparameter_edges()

    def arg_only_edges(self) -> List[Tuple[Any, Any, int]]:
        """
        Edges that bring arguments except hyperparameters into the algorithm.

        Returns:
            A list of 3-tuples, where the first element is the source, the second element is the
            target, and the third element is the order of the argument in the
            target's __call__ method.
        """
        parameters_mapping_proxy = inspect.signature(self.__call__).parameters
        return [
            (key, self, arg_order)
            for arg_order, key in enumerate(parameters_mapping_proxy.keys())
            if key != "self" and key not in collect_hyperparameters(self)
        ]

    def hyperparameter_edges(self) -> List[Tuple[Any, Any, int]]:
        """
        Edges that bring hyperparameters into the algorithm.

        Returns:
            A list of 3-tuples, where the first element is the source, the second element is the
            target, and the third element is the order of the argument in the
            target's __call__ method.
        """
        return [
            (key, self, arg_order)
            for arg_order, key in enumerate(collect_hyperparameters(type(self)).keys())
        ]

    def target_edges(self) -> List[Tuple[Any, Any, int]]:
        """
        Edges that distribute the output of the algorithm to the next algorithm in the pipeline.

        Returns:
            A list of 3-tuples, where the first element is the source, the second element is the
            target, and the third element is the order of the argument for the target.
        """
        return [(self, self.resource_name, 0)]
