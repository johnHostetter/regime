"""
This module defines the simple components expected and used for the flow of a Regime.
"""
from collections import namedtuple

# a resource (e.g., a dataset, a tensor, a module, a list, a configuration setting, or a device)
Resource = namedtuple(typename="Resource", field_names=("name", "value"))

# process (e.g., a function or a callable class)
Process = namedtuple(
    typename="Process", field_names=("name", "callable", "thread", "output")
)
