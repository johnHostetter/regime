"""
Implements a lightweight record of a process's callable, arguments, and output
for the Regime class.
"""

from typing import Any, Dict


class ComponentThread:
    """
    Carries a callable function for the Regime class along with the keyword
    arguments to invoke it with, an assigned name, and its output once it has
    run.

    Despite the name (kept for backward compatibility - nothing outside this
    module constructs or type-checks against it), this does not run on a
    separate thread: Regime.process_frontier() always calls
    `thread.function(**kwargs)` directly, synchronously, on the calling
    thread. Previously this class subclassed threading.Thread and implemented
    run()/join() for genuine concurrent execution, but nothing ever called
    start()/join() to use that machinery, so it was dead code - it paid
    threading.Thread.__init__'s overhead, and its own DeprecationWarning fired
    on every single process execution, on every run.
    """

    def __init__(self, function: callable, name: str = None, **kwargs):
        self.function = function
        self.kwargs: Dict[str, Any] = kwargs
        if name is None:
            self.name = self.graph_name = str(function)
        else:
            self.name, self.graph_name = name, "\n".join(name.split(" "))
        self.output = None

    def __str__(self) -> str:
        return self.graph_name
