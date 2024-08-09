"""
Test Regime functionality. The Regime class controls and facilitates the execution of functions
in a thread-safe manner. It is used to manage the flow of data between processes.
"""

import unittest
from pathlib import Path
from typing import Union, Dict, Any

import yaml
import torch
import igraph as ig

from regime import Regime, Resource, Node, hyperparameter
from tests.submodule import (
    ExampleClassB,
)  # class located here to showcase hierarchy

AVAILABLE_DEVICE = torch.device("cpu")  # CUDA out of memory error for these tests
# constant values used for testing
ALPHA: float = 0.1
BETA: float = 0.3
GAMMA: float = 0.7
FINAL_VALUE: int = 2025
FINAL_RESULT: Dict[str, Union[str, int]] = {
    "tests.test_regime.example_func": str(BETA),
    "example_d": FINAL_VALUE,
}


# classes used to demonstrate the Regime functionality
class ExampleClassA(Node):  # pylint: disable=too-few-public-methods
    """
    Example class A that uses the Node metaclass.
    """

    @hyperparameter("alpha")
    def __call__(
        self, input_data: torch.Tensor, alpha: float, device: torch.device
    ) -> float:
        return input_data.sum().item() + alpha


class ExampleClassC(Node):  # pylint: disable=too-few-public-methods
    """
    Example class C that uses the Node metaclass.
    """

    @hyperparameter("gamma")
    def __call__(self, example_a: float, example_b: float, gamma: float) -> float:
        return example_a + example_b + gamma


class ExampleClassD(Node):  # pylint: disable=too-few-public-methods
    """
    Example class D that uses the Node metaclass.
    """

    def __call__(self, example_c: float) -> int:
        return FINAL_VALUE


def example_func(example_b: float) -> str:
    """
    An example function to showcase functionality of the Regime class with a function.

    Args:
        example_b: A float value.

    Returns:
        A string representation of the float value.
    """
    return str(example_b)


class TestRegime(unittest.TestCase):
    """
    The self-organizing process can be thought as a Knowledge Base (KB) constructing another KB.
    However, it passes the relevant components needed to call the expert design process when
    it has finished, to conclude the construction of the KB.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.callables: Union[Node, callable] = [
            ExampleClassA(resource_name="example_a"),
            ExampleClassB(resource_name="example_b"),
            ExampleClassC(resource_name="example_c"),
            example_func,
        ]  # either a function or a Node object are allowed
        self.data = torch.zeros(10, 2)
        with open(
            Path(__file__).parent / "test_configuration.yaml", "r", encoding="utf8"
        ) as file:
            self.config = yaml.safe_load(file)

    def test_invalid_regime(self) -> None:
        """
        Test that if we create Regime object without any functions, then a ValueError should
        be thrown (as there is no point in creating it prematurely). The same should happen if we
        try to add a non-callable object to the Regime object.

        Returns:
            None
        """
        # trying to create Regime without any functions should raise a ValueError
        self.assertRaises(ValueError, Regime, callables={})

        # trying to create Regime with a non-callable object should raise a ValueError
        self.assertRaises(ValueError, Regime, callables={1})

    def test_add_callable(self) -> None:
        """
        We can add a single callable.

        Returns:
            None
        """
        regime = Regime(callables={lambda x: x})
        assert len(regime.graph.vs) == 1  # for the 1 callable
        assert len(regime.graph.es) == 0

    def test_add_callables(self) -> None:
        """
        We can add an iterable collection of callable objects (e.g., functions, classes).

        Returns:
            None
        """
        regime = Regime(callables=self.callables)
        assert len(regime.graph.vs) == len(self.callables)
        assert len(regime.graph.es) == 0  # we have not linked the functions yet
        # check that the required hyperparameters are as expected for the given callables
        # (i.e., the callables that have the 'hyperparameters' attribute)
        # value None means that the hyperparameter is not set
        assert regime.required_hyperparameters == {
            # follows the project structure
            "tests": {
                "submodule": {"ExampleClassB": {"beta": None}},
                "test_regime": {
                    "ExampleClassC": {"gamma": None},
                    "ExampleClassA": {"alpha": None},
                },
            }
        }  # the configuration file passed to Regime MUST contain the above hyperparameters

    def test_load_hyperparameters(self) -> None:
        """
        Test that required hyperparameters are loaded correctly from the configuration provided.

        Returns:
            None
        """
        regime = Regime(callables=self.callables)
        # create a config that satisfies the required hyperparameters
        defined_hyperparameters = regime.define_hyperparameters(
            configuration=self.config
        )
        assert defined_hyperparameters == {
            # follows the project structure
            "tests": {
                "submodule": {"ExampleClassB": {"beta": BETA}},
                "test_regime": {
                    "ExampleClassC": {"gamma": GAMMA},
                    "ExampleClassA": {"alpha": ALPHA},
                },
            }
        }  # the configuration file passed to Regime MUST contain the above hyperparameters

    def test_add_edges(self) -> None:
        """
        Test that adding several edges works as intended.

        Returns:
            None
        """
        regime = Regime(callables=self.callables)
        edges = [
            (self.callables[0], self.callables[2], 0),
            (self.callables[1], self.callables[2], 1),
        ]
        # create a config that satisfies the required hyperparameters
        # set up the Regime object with the configuration
        regime.setup(
            configuration=self.config, resources=None, edges=None, clean_up=False
        )
        # link the processes together
        regime.define_flow(edges, clean_up=False)
        assert len(regime.graph.vs) == len(self.callables) + len(regime.resources)
        assert len(regime.graph.es) == len(edges)

        edges = [
            (ExampleClassC, ExampleClassD, 0),
        ]
        self.assertRaises(ValueError, regime.define_flow, edges)

    def test_with_functions_that_make_no_progress(self) -> None:
        """
        If the Regime object has vertices that possess an attribute called 'function', but
        none of them make any progress, then the Regime procedure abruptly stops.

        Returns:
            None
        """

        def make_no_progress() -> None:
            """
            This function does nothing.

            Returns:
                None
            """
            return None

        def wait_for_no_progress(arg_1: Any, arg_2: Any) -> None:
            """
            This function waits for the two arguments to be printed.

            Args:
                arg_1: Any argument.
                arg_2: Any argument.

            Returns:
                None
            """
            print(arg_1, arg_2)

        make_no_progress_1 = make_no_progress_2 = make_no_progress

        callables = {
            make_no_progress_1,
            make_no_progress_2,
            wait_for_no_progress,
        }
        regime = Regime(callables=callables)
        edges = [
            (make_no_progress_1, wait_for_no_progress, 0),
            (make_no_progress_2, wait_for_no_progress, 1),
        ]
        regime.define_flow(edges)
        num_of_vertices, num_of_edges = len(callables), len(edges)
        assert len(regime.graph.vs) == num_of_vertices
        assert len(regime.graph.es) == num_of_edges
        assert regime.start() == {
            "tests.test_regime.wait_for_no_progress": None
        }  # do nothing

    def test_add_invalid_source_vertex(self) -> None:
        """
        Test that adding an edge with an invalid source vertex raises an error.

        Returns:
            None
        """
        callables = {
            ExampleClassB,
            ExampleClassC,
        }
        regime = Regime(callables=callables)
        edges = [
            (None, ExampleClassC, 0),
            (ExampleClassA, ExampleClassC, 1),
            ("gamma", ExampleClassC, 2),
        ]
        # assert regime.link_functions(edges) throws a ValueError because of the None vertex
        self.assertRaises(ValueError, regime.define_flow, edges)
        # assert regime.link_functions(edges) throws a ValueError because it is missing the
        # ExampleClassA in its 'functions' argument
        self.assertRaises(ValueError, regime.define_flow, edges[1:])

    def test_add_invalid_target_vertex(self) -> None:
        """
        Test that adding an edge with an invalid target vertex raises an error.

        Returns:
            None
        """
        callables = {
            ExampleClassA,
            ExampleClassB,
        }
        regime = Regime(callables=callables)
        edges = [
            (ExampleClassA, None, 0),
            (ExampleClassB, ExampleClassC, 1),
            ("gamma", ExampleClassC, 2),
        ]
        # assert regime.link_functions(edges) throws a ValueError because of the None vertex
        self.assertRaises(ValueError, regime.define_flow, edges)
        # assert regime.link_functions(edges) throws a ValueError because it is missing the
        # WM in its 'functions' argument
        self.assertRaises(ValueError, regime.define_flow, edges[1:])

    def test_add_input_data(self) -> None:
        """
        Test adding a special vertex to store the input data. The special vertex's value is passed
        as an argument to functions that rely upon input data.

        Returns:
            None
        """
        regime = Regime(callables=self.callables)
        edges = [
            (self.callables[0], self.callables[2], 0),
            (self.callables[1], self.callables[2], 1),
            ("gamma", self.callables[2], 2),
        ]
        regime.setup(
            configuration=self.config,
            resources={Resource(name="input", value=self.data)},
            edges=edges,
            clean_up=False,
        )
        assert len(regime.graph.vs) == len(self.callables) + len(
            regime.resources
        )  # (incl. data vertex)
        assert len(regime.graph.es) == len(edges)

        # test that we can decide to add more edges to the regime's workflow
        more_edges = [
            ("input", self.callables[0], 0),
            ("input", self.callables[1], 0),
        ]
        regime.define_flow(more_edges)
        assert len(regime.graph.es) == len(edges) + len(more_edges)

    def test_get_kwargs(self) -> None:
        """
        Test that the keyword arguments are as expected in Regime.

        Returns:
            None
        """
        regime = Regime(callables=self.callables)
        edges = self.callables[0].edges() + self.callables[1].edges()
        regime.setup(
            configuration=self.config,
            resources={
                Resource(name="input_data", value=self.data),
                Resource(name="device", value=AVAILABLE_DEVICE),
            },
            edges=edges,
            clean_up=False,
        )
        assert len(regime.graph.vs) == len(self.callables) + len(
            regime.resources
        )  # (incl. data & config vertices)
        assert len(regime.graph.es) == len(edges)

        more_edges = self.callables[2].edges()
        regime.define_flow(more_edges)
        assert len(regime.graph.es) == len(edges) + len(
            more_edges
        )  # edges added to existing

        # find the vertex for this function & its predecessors
        target_vertex = regime.graph.vs.find(callable_eq=self.callables[2])

        expected_kwargs = {
            "example_a": None,
            "example_b": None,
            "gamma": GAMMA,
        }
        assert regime.get_keyword_arguments(target_vertex) == expected_kwargs

    def test_unlinked_start(self) -> None:
        """
        Test that the self-organizing process raises an error if it is started without any
        vertices that are linked.

        Returns:
            None
        """
        regime = Regime(callables=self.callables)
        self.assertRaises(ValueError, regime.start)

    def test_start(self) -> None:
        """
        Test a verbose definition of a self-organizing process (i.e., no shortcut method call).

        Returns:
            KnowledgeBase
        """
        class_d = ExampleClassD(resource_name="example_d")
        regime = Regime(
            callables=self.callables + [class_d],
            resources={
                Resource(name="input_data", value=self.data),
                Resource(name="device", value=AVAILABLE_DEVICE),
            },
            verbose=True,
        )
        regime.setup(
            configuration=self.config,
            edges=self.callables[0].edges()
            + self.callables[1].edges()
            + self.callables[2].edges()
            + class_d.edges()
            + [
                ["example_b", example_func, 0]
            ],  # example_func is a no-op here, but used to showcase the feature
        )

        result: Dict[str, Union[str, float]] = regime.start()

        # --- test info flowed properly from input_data to ExampleClassA ---
        actual_kwargs = regime.get_keyword_arguments(
            regime.get_vertex(self.callables[0])
        )
        expected_kwargs = {
            "input_data": self.data,
            "alpha": ALPHA,
            "device": AVAILABLE_DEVICE,
        }
        assert actual_kwargs == expected_kwargs

        # --- test info flowed properly from input_data to ExampleClassB ---
        actual_kwargs = regime.get_keyword_arguments(
            regime.get_vertex(self.callables[1])
        )
        expected_kwargs = {
            "input_data": self.data,
            "beta": BETA,
            "device": AVAILABLE_DEVICE,
        }
        assert actual_kwargs == expected_kwargs

        # --- test info flowed properly from ExampleClassB to example_func (i.e., broadcast) ---
        output_b = regime.graph.vs.find(callable_eq=self.callables[1])["output"]
        actual_kwargs = regime.get_keyword_arguments(regime.get_vertex(example_func))
        expected_kwargs = {"example_b": output_b}
        assert actual_kwargs == expected_kwargs

        # --- test info flowed properly from ExampleClassA and ExampleClassB to ExampleClassC ---
        output_a = regime.graph.vs.find(callable_eq=self.callables[0])["output"]
        actual_kwargs = regime.get_keyword_arguments(
            regime.get_vertex(self.callables[2])
        )
        expected_kwargs = {"example_a": output_a, "example_b": output_b, "gamma": GAMMA}
        print(actual_kwargs)
        print(expected_kwargs)
        assert actual_kwargs == expected_kwargs

        # --- test info flowed properly from ExampleClassC to ExampleClassD ---
        output_c = regime.graph.vs.find(callable_eq=self.callables[2])["output"]
        actual_kwargs = regime.get_keyword_arguments(regime.get_vertex(class_d))
        expected_kwargs = {"example_c": output_c}
        assert actual_kwargs == expected_kwargs

        # --- check that the final output is as expected ---
        assert result == FINAL_RESULT

        # --- demonstrate plotting of the Regime ---
        layout = regime.graph.layout("sugiyama")  # a hierarchical layout
        visual_style = {
            "margin": 60,  # pads whitespace around the plot
            "edge_width": 2.0,
            "edge_curved": 0.1,
            "edge_arrow_size": 2.0,
            "edge_arrow_width": 0.5,
            "vertex_label_dist": 2.5,  # distance of label from the vertex
            "vertex_label_color": "#029e73",
            "vertex_label_angle": 3.14,
            "vertex_label_size": 10,
            "vertex_size": 20,
            "vertex_color": [
                "#0173b2" if vertex["type"] == "process" else "#de8f05"
                for vertex in regime.graph.vs
            ],
            "vertex_shape": [
                "circle" if vertex["type"] == "process" else "rectangle"
                for vertex in regime.graph.vs
            ],
        }  # margin controls the amount of padding around the plot
        pretty_names = [name.split(".")[-1] for name in regime.graph.vs["name"]]
        regime.graph.vs["label"] = pretty_names
        for file_extension in ["png", "pdf"]:
            # save the plot as a PNG and PDF file
            ig.plot(
                regime.graph,
                target=str(Path(__file__).parent / f"test_regime.{file_extension}"),
                layout=layout,  # specify the layout of the vertices
                autocurve=True,  # curved edges
                **visual_style,  # apply the visual style
            )
