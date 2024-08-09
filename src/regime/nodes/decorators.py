"""
Decorators for working with Regime Nodes.
"""


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
