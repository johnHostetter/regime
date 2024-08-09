"""
Utilities for working with Regime code.
"""


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
