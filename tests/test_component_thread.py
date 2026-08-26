"""
Test the ComponentThread class - a plain (non-threaded) record of a
callable, its keyword arguments, and its output. See threads.py's module
docstring for why this no longer subclasses threading.Thread.
"""

import unittest

from regime.flow.threads import ComponentThread


def _example_function(**kwargs) -> str:
    """A stand-in callable to wrap in a ComponentThread for these tests."""
    return "called"


class TestComponentThread(unittest.TestCase):
    """
    Test the ComponentThread class.
    """

    def test_name_given(self) -> None:
        """
        An explicit name is stored as-is in .name, and reflected (with spaces
        replaced by newlines) in .graph_name and __str__.
        """
        name = "test name"
        thread = ComponentThread(function=_example_function, name=name)
        self.assertEqual(thread.name, name)
        self.assertEqual(thread.graph_name, "test\nname")
        self.assertEqual(str(thread), "test\nname")

    def test_name_defaults_to_str_of_the_function(self) -> None:
        """
        Without an explicit name, .name and .graph_name both default to
        str(function) (a single value, not name-with-spaces-replaced, since
        there is no separate "name" to derive graph_name from).
        """
        thread = ComponentThread(function=_example_function)
        self.assertEqual(thread.name, str(_example_function))
        self.assertEqual(thread.graph_name, str(_example_function))
        self.assertEqual(str(thread), str(_example_function))

    def test_output_defaults_to_none_until_set(self) -> None:
        """
        .output is not computed by construction - Regime.process_frontier()
        calls the function itself and assigns the result to .output
        afterward.
        """
        thread = ComponentThread(function=_example_function)
        self.assertIsNone(thread.output)
        thread.output = _example_function()
        self.assertEqual(thread.output, "called")

    def test_kwargs_are_stored(self) -> None:
        """
        Keyword arguments passed to the constructor (beyond function/name)
        are captured in .kwargs, for the caller to later pass to .function.
        """
        thread = ComponentThread(function=_example_function, a=1, b=2)
        self.assertEqual(thread.kwargs, {"a": 1, "b": 2})

    def test_does_not_run_on_a_separate_thread(self) -> None:
        """
        Regression guard: ComponentThread must not be a threading.Thread
        subclass (or otherwise offer start()/join()) - Regime always invokes
        .function directly and synchronously; anything suggesting real
        concurrency here would be misleading.
        """
        thread = ComponentThread(function=_example_function)
        self.assertFalse(hasattr(thread, "start"))
        self.assertFalse(hasattr(thread, "join"))
        self.assertFalse(hasattr(thread, "run"))


if __name__ == "__main__":
    unittest.main()
