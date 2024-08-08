"""
Defines the metaclass for Regime Nodes. This metaclass automatically collects hyperparameters from
the class (assuming it has been decorated with the hyperparameter decorator) and stores them in a
dictionary.
"""
from abc import ABC


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
