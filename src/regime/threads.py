"""
Implements the necessary threading logic for the soft.computing.organize.SelfOrganize class.
"""

import threading


class ComponentThread(threading.Thread):
    """
    This class contains the necessary logic to prepare a callable function
    for the Regime class. It allows for the callable function to be
    partially initialized with keyword arguments, assign a name to it, as well
    as saves the function's output in self.output when it finishes.
    """

    def __init__(self, function: callable, name: str = None, **kwargs):
        super().__init__()
        self.function = function
        self.kwargs = kwargs
        if name is None:
            self.name = self.graph_name = str(function)
        else:
            self.name, self.graph_name = name, "\n".join(name.split(" "))
        self.output = None
        self.exception = None

    def __str__(self) -> str:
        return self.graph_name

    def run(self) -> None:
        """
        The function that is executed in a new thread.

        Returns:
            None
        """
        try:
            self.output = self.function(**self.kwargs)
        # pylint: disable=locally-disabled, broad-exception-caught
        except BaseException as exception:
            self.exception = exception

    def join(self, timeout=None) -> None:
        threading.Thread.join(self, timeout)
        # re-raise any exception thrown in the thread
        if self.exception:
            raise self.exception
