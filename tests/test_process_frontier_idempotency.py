"""
White-box tests for Regime.process_frontier()'s idempotency guard.

Regime.start() drives a frontier (a plain list used as a stack) where a
process vertex with N predecessors is appended to the frontier once per
predecessor that finishes (see process_frontier()'s successors_vertices
expansion). If two or more of those N completions happen before any of the
resulting frontier entries for that vertex is popped, every one of those
entries independently observes all predecessors ready and calls the vertex's
function again - a silent double (or more) execution. complete_process_vertices
exists specifically to prevent this (a set of already-executed vertices,
checked by process_frontier() before it will run a vertex's function again).

These tests bypass Regime.start()'s traversal order entirely - that order
depends on igraph/set iteration internals that calling code should not rely
on - and instead call process_frontier() directly. This makes the tests a
deterministic proof of the guard, rather than incidental behavior of one
particular graph shape or traversal order. A black-box integration test using
the real Regime.start() traversal is included at the end as a companion,
not a replacement.
"""

import unittest
from typing import Any, Dict, List, Tuple

import igraph

from regime import Regime


def _make_counting_function(name: str, return_value: Any = None):
    """
    Build a function that records every call's kwargs and returns
    `return_value`, so tests can assert exactly how many times - and with
    what arguments - it was invoked.

    Args:
        name: Assigned as the function's __name__, so vertices built from
            multiple such functions have distinct, readable names in the
            Regime graph instead of all sharing the factory's own name.
        return_value: The value the function returns on every call.

    Returns:
        The callable, with a `.calls` attribute (a list of the kwargs dicts
        it was called with, in call order).
    """
    calls: List[Dict[str, Any]] = []

    def _function(**kwargs) -> Any:
        calls.append(kwargs)
        return return_value

    _function.__name__ = name
    _function.calls = calls
    return _function


class TestProcessFrontierIdempotency(unittest.TestCase):
    """
    Covers Regime.process_frontier()'s complete_process_vertices guard
    directly, independent of Regime.start()'s traversal order.
    """

    def _build_ready_diamond(
        self,
    ) -> Tuple[Regime, Any, igraph.Vertex]:
        """
        Build a_fn -> c_fn, b_fn -> c_fn (a_fn and b_fn are both roots feeding
        a single downstream c_fn), then drive a_fn and b_fn to completion
        directly - only c_fn's idempotency is under test.

        Returns:
            A 3-tuple of (regime, c_fn, c_vertex).
        """
        a_fn = _make_counting_function("a_fn", return_value=1)
        b_fn = _make_counting_function("b_fn", return_value=2)
        c_fn = _make_counting_function("c_fn", return_value=3)

        regime = Regime(callables={a_fn, b_fn, c_fn})
        regime.define_flow(edges=[(a_fn, c_fn, 0), (b_fn, c_fn, 1)])

        regime.process_frontier(regime.get_vertex(a_fn), [], None)
        regime.process_frontier(regime.get_vertex(b_fn), [], None)
        self.assertEqual(len(a_fn.calls), 1)
        self.assertEqual(len(b_fn.calls), 1)

        return regime, c_fn, regime.get_vertex(c_fn)

    def test_executes_exactly_once_when_predecessors_ready(self) -> None:
        """
        Sanity check: a vertex whose predecessors are all ready executes
        exactly once on a single process_frontier() call, and is recorded as
        complete.
        """
        regime, c_fn, c_vertex = self._build_ready_diamond()
        regime.process_frontier(c_vertex, [], None)
        self.assertEqual(len(c_fn.calls), 1)
        self.assertIn(c_vertex, regime.complete_process_vertices)

    def test_second_call_does_not_re_execute(self) -> None:
        """
        The exact shape of the bug: the same vertex is handed to
        process_frontier() twice, as would happen if it were appended to the
        frontier list twice (once per predecessor completion) before either
        entry was popped. The underlying function must only run once, and the
        second (skipped) call must not append successors again or touch
        `thread`.
        """
        regime, c_fn, c_vertex = self._build_ready_diamond()
        regime.process_frontier(c_vertex, [], None)

        frontier_vertices, thread = regime.process_frontier(c_vertex, [], None)

        self.assertEqual(len(c_fn.calls), 1)
        self.assertEqual(frontier_vertices, [])
        self.assertIsNone(thread)

    def test_repeated_calls_still_execute_only_once(self) -> None:
        """
        The guard holds no matter how many times the same already-complete
        vertex is (re-)handed to process_frontier() - not just a "second
        time" special case.
        """
        regime, c_fn, c_vertex = self._build_ready_diamond()
        for _ in range(5):
            regime.process_frontier(c_vertex, [], None)
        self.assertEqual(len(c_fn.calls), 1)

    def test_complete_process_vertices_tracks_exactly_the_executed_vertices(
        self,
    ) -> None:
        """
        complete_process_vertices is a set - repeated completion attempts on
        an already-complete vertex must not grow it beyond the vertices that
        actually ran (a_fn, b_fn, c_fn - three total, not more).
        """
        regime, _, c_vertex = self._build_ready_diamond()
        regime.process_frontier(c_vertex, [], None)
        regime.process_frontier(c_vertex, [], None)
        self.assertEqual(len(regime.complete_process_vertices), 3)

    def test_does_not_execute_when_a_predecessor_is_still_missing(self) -> None:
        """
        Regression guard for the other half of process_frontier()'s
        condition: the new idempotency guard must not mask, or interact badly
        with, the pre-existing "not all predecessors are ready yet" branch.
        """
        a_fn = _make_counting_function("a_fn", return_value=1)
        b_fn = _make_counting_function("b_fn", return_value=2)
        c_fn = _make_counting_function("c_fn", return_value=3)
        regime = Regime(callables={a_fn, b_fn, c_fn})
        regime.define_flow(edges=[(a_fn, c_fn, 0), (b_fn, c_fn, 1)])

        # only a_fn has run - b_fn has not, so c_fn's predecessors are not all
        # ready
        regime.process_frontier(regime.get_vertex(a_fn), [], None)

        c_vertex = regime.get_vertex(c_fn)
        frontier_vertices, thread = regime.process_frontier(c_vertex, [], None)

        self.assertEqual(len(c_fn.calls), 0)
        self.assertNotIn(c_vertex, regime.complete_process_vertices)
        self.assertEqual(frontier_vertices, [])
        self.assertIsNone(thread)

    def test_isolated_vertex_is_never_marked_complete(self) -> None:
        """
        A disconnected (degree 0) vertex is skipped entirely by
        process_frontier() (pre-existing behavior, unrelated to this fix) -
        confirm the new guard's ordering (degree check first, short-circuiting
        before the complete_process_vertices check) preserves that, rather
        than an isolated vertex ever being considered for completion at all.
        """
        lonely_fn = _make_counting_function("lonely_fn")
        regime = Regime(callables={lonely_fn})
        vertex = regime.get_vertex(lonely_fn)

        frontier_vertices, thread = regime.process_frontier(vertex, [], None)

        self.assertEqual(len(lonely_fn.calls), 0)
        self.assertNotIn(vertex, regime.complete_process_vertices)
        self.assertEqual(frontier_vertices, [])
        self.assertIsNone(thread)


class TestRegimeStartDiamondIntegration(unittest.TestCase):
    """
    Black-box companion to TestProcessFrontierIdempotency: exercises the same
    a_fn -> c_fn, b_fn -> c_fn diamond shape through the real, public
    Regime.start() traversal, confirming the fix holds for actual end-to-end
    usage and not just direct process_frontier() calls.
    """

    def test_converging_process_runs_exactly_once(self) -> None:
        """
        A process with two predecessors, run through the real Regime.start()
        traversal, must execute exactly once - regardless of the internal
        pop()/successors() ordering that determines how many times it is
        appended to the frontier before completing.
        """
        a_fn = _make_counting_function("a_fn", return_value=1)
        b_fn = _make_counting_function("b_fn", return_value=2)
        c_fn = _make_counting_function("c_fn", return_value=3)

        regime = Regime(callables={a_fn, b_fn, c_fn})
        regime.define_flow(edges=[(a_fn, c_fn, 0), (b_fn, c_fn, 1)])

        result = regime.start()

        self.assertEqual(len(a_fn.calls), 1)
        self.assertEqual(len(b_fn.calls), 1)
        self.assertEqual(len(c_fn.calls), 1)
        # vertex names are f"{__module__}.{__name__}" (see Regime.to_processes())
        self.assertEqual(result, {f"{c_fn.__module__}.c_fn": 3})


if __name__ == "__main__":
    unittest.main()
