"""
Test the ComponentThread class.
"""

import unittest

from regime.threads import ComponentThread


class TestComponentThread(unittest.TestCase):
    """
    Test the ComponentThread class.
    """

    def return_function_not_implemented_error(self) -> None:
        """
        This function is used to test the ComponentThread class. It raises a NotImplementedError.

        Returns:
            None
        """
        raise NotImplementedError("This function is not implemented.")

    def test_name_given(self) -> None:
        """
        Test that the name of the thread is correctly set.
        """
        name = "test_name"
        thread = ComponentThread(
            function=self.return_function_not_implemented_error, name=name
        )
        self.assertEqual(thread.name, name)
        self.assertEqual(thread.graph_name, name)
        self.assertEqual(str(thread), name)

    def test_run_fails(self) -> None:
        """
        Test that the thread fails to run if the function is not set.
        """
        thread = ComponentThread(function=self.return_function_not_implemented_error)
        # no exception should be stored in the thread yet
        self.assertTrue(thread.exception is None)
        # thread will raise an exception when run, but it will be caught and stored in the thread
        thread.run()
        # a function that raises an Exception will be caught and stored in the thread
        self.assertTrue(isinstance(thread.exception, NotImplementedError))
        thread.start()  # this will store an exception, but we can re-raise it by calling join
        self.assertRaises(NotImplementedError, thread.join)
