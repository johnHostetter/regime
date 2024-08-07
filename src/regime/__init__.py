"""
Implements functions and classes responsible for the design of soft computing solutions,
such as the Regime class.
"""

from collections import namedtuple
from typing import List, Tuple, Any, Union, Set, Iterable, Dict, OrderedDict

import igraph
from rough.decisions import RoughDecisions

from regime.threads import ComponentThread
from regime.utils import HyperparameterMeta, make_hyperparameters_dict, merge_dicts

# a resource (e.g., a dataset, a tensor, a module, a list, a configuration setting, or a device)
Resource = namedtuple(typename="Resource", field_names=("name", "value"))

# process (e.g., a function or a callable class)
Process = namedtuple(
    typename="Process", field_names=("name", "callable", "thread", "output")
)

# edge with an order argument for the target
EdgeWithOrder = namedtuple(typename="Edge", field_names=("source", "target", "order"))

# edge with a keyword argument for the target
EdgeWithKeyword = namedtuple(
    typename="Edge", field_names=("source", "target", "keyword")
)


class Regime(RoughDecisions):  # inherit from RoughDecisions to have access to functionality
    """
    The Regime class facilitates the convenient design of self-organizing solutions
    by adding callable function references to a RoughDecisions graph as vertices, and using
    the edges that connect the vertices (i.e., functions) to determine the flow of
    data/information from function to function.
    """

    # (0) receive all processes as a list of functions
    # (1) add all processes (functions) as vertices
    # (1.5) COLLECT all configurations needed for the functions, as well as RESOURCES
    # (2) receive all resources as a list of Resource tuples
    # (3) add all resources as vertices
    # (4) receive all edges as a list of EdgeWithOrder tuples (or EdgeWithKeyword tuples)
    # (5) add all edges to the graph
    # (6) CHECK if all resources are present in the graph (if not, use the configuration settings)
    # (7) CHECK the flow of data from function to function
    # (8) remove any isolated vertices
    # (9) DOUBLE CHECK if all is in order
    # (10) start the self-organizing process

    def __init__(
        self,
        callables: Set[callable],
        resources: Union[None, Set[Resource]] = None,  # resources are optional
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.verbose = verbose
        self.complete_process_vertices: Set[igraph.Vertex] = set()

        # the set of callables must be non-empty
        if len(callables) == 0:
            raise ValueError("No callables have been provided.")

        # first make it a list of processes
        self.processes, self.required_hyperparameters = self.to_processes(
            callables=callables
        )

        # then add the processes to the actual graph as isolated vertices
        self.add_processes()

        # now a required hyperparameters dictionary has been created from specifying the processes
        # it must be populated with values from a provided configuration settings in the follow-up

        # now we can add the resources to the graph
        self.resources: Set[Resource] = set()
        if resources is not None:
            self.add_resources(resources=resources)

    @property
    def process_vertices(self) -> List[igraph.Vertex]:
        """
        Get the process vertices from the Regime graph.

        Returns:
            A list of process vertices.
        """
        return [v for v in self.graph.vs if v["type"] == "process"]

    @property
    def unprepared_resource_vertices(self) -> List[igraph.Vertex]:
        """
        Get the resource vertices that have not yet been prepared in the Regime's execution
        (i.e., they have no output).

        Returns:
            A list of resource vertices.
        """
        return [
            v for v in self.graph.vs if v["type"] == "resource" and v["output"] is None
        ]

    @staticmethod
    def to_processes(callables: Iterable[callable]) -> Tuple[List[Process], dict]:
        """
        A Process is a callable and threadable object. Add the callables that should be permitted
        for use by the Regime.start() method.

        Args:
            callables: A list of classes or functions that are allowed to be used.

        Returns:
            A tuple of two elements: a list of Process objects and a dictionary of required
            hyperparameters for those processes.
        """
        processes: List[Process] = []
        required_hyperparameters: dict = {}
        for _callable in callables:
            # the output attribute will later be changed once the ComponentThread executes
            if callable(_callable):
                # e.g., name = "fuzzy_ml.clustering.empirical.find_empirical_fuzzy_sets"
                name: str = f"{_callable.__module__}.{_callable.__name__}"
                if isinstance(_callable, HyperparameterMeta):
                    # find the hyperparameters required to be specified in configuration settings
                    required_hyperparameters = merge_dicts(
                        existing=required_hyperparameters,
                        new=make_hyperparameters_dict(_callable),
                    )
                processes.append(
                    Process(
                        name=name,
                        callable=_callable,
                        thread=None,
                        output=None,
                    )
                )
            else:
                raise ValueError(
                    f"Callable {_callable} is not a function or class of HyperparameterMeta."
                )
        return processes, required_hyperparameters

    def add_processes(self) -> None:
        """
        Add the processes to the Regime's graph as isolated vertices.

        Returns:
            None
        """
        self.graph.add_vertices(len(self.processes))
        self.graph.vs["type"] = "process"
        self.graph.vs["name"] = [process.name for process in self.processes]
        self.graph.vs["callable"] = [process.callable for process in self.processes]
        self.graph.vs["output"] = [process.output for process in self.processes]
        self.graph.vs["thread"] = [process.thread for process in self.processes]

    def add_resources(self, resources: Set[Resource]) -> None:
        """
        Add resources as isolated vertices in the Regime's graph. This is encouraged if a priori
        resources are to be used during the Regime's execution.

        Note 1: If a Resource is to be created from a Process, it is not required to add it.

        Note 2: Not all Resource objects are required to be used in the Regime's execution, but they
        are available for use if needed. If a resource is not used, it will not be modified.

        Note 3: If a resource is used, it will be modified by the process that uses it.

        Args:
            resources: A set of resources to add to the graph.

        Returns:
            None
        """
        for resource in resources:
            if resource in self.resources or self.graph.vs.select(
                name_eq=resource.name
            ):
                raise ValueError(f"Resource {resource.name} already exists.")
            self.resources.add(resource)
            self.graph.add_vertex(
                type="resource",
                name=resource.name,
                function=None,
                output=resource.value,
                thread=None,
            )

    def define_hyperparameters(
        self,
        configuration: OrderedDict,
        remaining_hyperparameters: Union[None, dict] = None,
    ) -> dict:
        """
        Parses the configuration settings to find the necessary hyperparameters for the processes
        that have been added to the Regime graph.

        Args:
            configuration: The configuration settings to use.
            remaining_hyperparameters: The required hyperparameters for the processes that have
            yet to be defined. If None, the required hyperparameters will be pulled from the
            Regime object. Otherwise, a recursive call has taken place and the nested dictionary
            is being traversed. Default is None.

        Returns:
            A dictionary of the necessary hyperparameters populated with their values.
        """
        if remaining_hyperparameters is None:
            remaining_hyperparameters = self.required_hyperparameters

        # find all None values in remaining_hyperparameters and assign them from the config
        keys = remaining_hyperparameters.keys()
        for key in keys:
            if remaining_hyperparameters[key] is None:
                remaining_hyperparameters[key] = configuration[key]
                # add  the hyperparameter (with its value) as a resource for the Regime
                self.add_resources(
                    resources={Resource(name=key, value=configuration[key])}
                )
            else:
                remaining_hyperparameters[key] = self.define_hyperparameters(
                    configuration=configuration[key],
                    remaining_hyperparameters=remaining_hyperparameters[key],
                )
        return remaining_hyperparameters

    def setup(
        self,
        configuration: Union[None, OrderedDict] = None,
        resources: Union[None, Set[Resource]] = None,  # resources are optional
        edges: Union[
            None, List[Union[EdgeWithOrder, EdgeWithKeyword]]
        ] = None,  # optional edges
        clean_up: bool = True,
    ) -> None:
        """
        Set up the Regime graph with the processes, resources, and edges that have been
        defined in the Regime object. Most of this is optional, as the Regime object can be
        set up in stages.

        Args:
            configuration: The configuration settings - must define all required hyperparameters;
            optional if no hyperparameters are required or are already defined.
            resources: Optional resources to add to the graph.
            edges: Optional edges to add to the graph.
            clean_up: Whether to remove any isolated vertices from the graph after adding the edges.

        Returns:
            None
        """
        # why is everything optional? --> We can set up the Regime graph in stages

        # define the hyperparameters' values for the processes
        if configuration is not None:
            self.define_hyperparameters(configuration=configuration)

        # add the resources to the graph (check for name conflicts w/ hyperparameters)
        if resources is not None:
            self.add_resources(resources=resources)

        if edges is not None:
            self.define_flow(edges=edges, clean_up=clean_up)

    def define_flow(
        self, edges: List[Tuple[Any, Any, Union[int, str]]], clean_up: bool = True
    ) -> None:
        """
        Links callables in the Regime's graph (via RoughDecisions) if they exist as vertices.

        This allows the transfer of inputs and outputs to easily occur later on.

        Args:
            edges: An iterable collection called edges, where each element in the collection is
            a 3-tuple such that it follows the form (source, target, arg_order/keyword_argument);
            this is called 'arg_specifier' (int or str).
            clean_up: Whether to remove any isolated vertices from the graph after adding the edges.
            Default is True.

            However, this method expects 'source' and 'target' to be referencing a
            callable function reference.

            The arg_specifier in a 3-tuple for an edge resolves the following dilemma; let's say
            Wang-Mendel expects CLIP and ECM to feed into it, and expects 2 arguments. Which output
            from which function goes where? The arg_specifier resolves this by specifying this
            order. Alternatively, a keyword argument can be used to specify the argument name
            (preferred).

        Returns:
            None
        """
        for source, target, arg_specifier in edges:
            source_vertex: igraph.Vertex = self.get_vertex(source)
            target_vertex: igraph.Vertex = self.get_vertex(target)
            if self.verbose:
                print(
                    f"Adding an edge between {source_vertex['name']} "
                    f"and {target_vertex['name']} "
                    f"to the Regime graph... ",
                    end="",
                    flush=True,
                )
            if (
                len(self.graph.edge_attributes()) > 0
                and len(  # no edge attributes means no edges
                    list(
                        self.graph.es.select(
                            _source_eq=source_vertex.index,
                            _target_eq=target_vertex.index,
                            arg_eq=arg_specifier,
                        )
                    )
                )
                > 0
            ):
                raise ValueError(
                    f"An edge already exists between {source_vertex['function']} "
                    f"and {target_vertex['function']}."
                )
            self.graph.add_edge(source_vertex, target_vertex, arg=arg_specifier)
            if self.verbose:
                print("Done.")

        # clean up the graph by removing any isolated vertices
        if clean_up:
            self.graph.delete_vertices(
                [v.index for v in self.graph.vs if v.degree() == 0]
            )

    def get_vertex(self, vertex_reference: Union[str, callable]) -> igraph.Vertex:
        """
        Get a vertex from the graph by its name or callable function reference. If it is a string,
        but does not yet exist, then we assume it is a resource from an intermediate process and
        create it.

        Args:
            vertex_reference: The name of the vertex or the callable function reference.

        Returns:
            The vertex object from the graph.
        """
        if isinstance(
            vertex_reference, str
        ):  # find the vertex by name, most likely resource
            if len(self.graph.vs.select(name_eq=vertex_reference)) == 0:
                # this is a new resource that does not yet exist
                # go ahead and make it - it may or may not be used later
                target_resource = Resource(
                    name=vertex_reference, value=None
                )  # no value yet
                self.add_resources(resources={target_resource})
            target_vertex: igraph.Vertex = self.graph.vs.find(name_eq=vertex_reference)
        else:  # find the vertex by callable, most likely process
            print(vertex_reference)
            target_vertex: igraph.Vertex = self.graph.vs.find(
                callable_eq=vertex_reference
            )
        return target_vertex

    @staticmethod
    def get_keyword_arguments(
        vertex: igraph.Vertex,
    ) -> Dict[str, Any]:
        """
        Given a vertex from the Regime graph, find the keyword arguments that are required for the
        vertex's callable to execute by referencing the outputs of the predecessors_vertices.

        Args:
            vertex: The vertex from the Regime graph.

        Returns:
            A dictionary of keyword arguments, where the keys are the names of the arguments,
            and the values are the outputs of the predecessors_vertices.
        """
        return {v["name"]: v["output"] for v in vertex.predecessors()}

    def start(self) -> Dict[str, Any]:
        """
        Run through the Regime's workflow with the processes and resources that have been linked.

        Returns:
            A dictionary mapping the names of the processes/resources to their outputs/values.
        """
        all_pending_vertices: List[igraph.Vertex] = (
            self.process_vertices + self.unprepared_resource_vertices
        )

        # create a subgraph consisting only of process vertices to find which vertices have no
        # incoming edges (i.e., no process predecessors - they wait for nothing to finish)
        process_subgraph: igraph.Graph = self.graph.induced_subgraph(
            all_pending_vertices
        )

        # find the vertices that have no incoming edges
        isolated_subgraph_vertices: List[igraph.Vertex] = [
            v for v in process_subgraph.vs if v.indegree() == 0
        ]

        # find the isolated vertices in the original graph
        isolated_vertices: List[igraph.Vertex] = [
            self.graph.vs.find(name_eq=v["name"]) for v in isolated_subgraph_vertices
        ]

        frontier_vertices, thread = isolated_vertices, None

        # note: functions must be unique or the below won't work (e.g., no 2 calls to 'train')
        while True:
            if len(frontier_vertices) == 0:
                break
            frontier_vertex = frontier_vertices.pop()
            if frontier_vertex["type"] == "resource":
                if frontier_vertex["output"] is not None:
                    # if the resource has output, then it is already complete
                    # expand the frontier to include the successors of the resource
                    frontier_vertices += frontier_vertex.successors()
                    continue  # skip the rest of the loop
                raise ValueError(
                    "Somehow a resource has been reached with no output (check Regime's flow)."
                )
            if frontier_vertex["type"] == "process":
                frontier_vertices, thread = self.process_frontier(
                    frontier_vertex, frontier_vertices, thread
                )

        if thread is not None:
            # return the output of all the processes at the end of workflow
            return {
                v["name"]: v["output"]
                for v in self.graph.vs
                if v.degree(mode="out") == 0
            }

        raise ValueError(
            "Last referenced thread is None. It should be the last executed function in the Regime."
        )

    def process_frontier(
        self, frontier_vertex, frontier_vertices, thread
    ) -> Tuple[List[igraph.Vertex], ComponentThread]:
        """
        Execute a process vertex (i.e., frontier_vertex) in the Regime graph.

        Args:
            frontier_vertex: The vertex to process.
            frontier_vertices: The list of vertices that are part of the frontier.
            thread: The thread to execute the process.

        Returns:
            A tuple of the updated frontier vertices and the thread that executed the process
        """
        function: callable = frontier_vertex["callable"]
        predecessors_vertices: igraph.VertexSeq = frontier_vertex.predecessors()
        if frontier_vertex.degree() > 0:  # only consider connected functions
            if len(predecessors_vertices) == 0 or all(  # no predecessors are fine or
                source_vertex["output"]
                is not None  # for all predecessors, we have output
                for source_vertex in predecessors_vertices
            ):
                # get the keyword arguments for this function
                kwargs: Dict[str, Any] = {
                    v["name"]: v["output"] for v in predecessors_vertices
                }
                if isinstance(function, HyperparameterMeta):
                    function = function()
                thread: ComponentThread = ComponentThread(function, **kwargs)
                frontier_vertex["thread"] = thread  # keep a reference to the thread
                thread.output = thread.function(**kwargs)
                # thread.start()
                # # thread.join()
                # retrieve thread output
                self.graph.vs.find(thread_eq=thread)["output"] = thread.output
                self.complete_process_vertices.add(frontier_vertex)
                # update any resources that have been modified by the function
                successors_vertices = frontier_vertex.successors()
                for successor in successors_vertices:
                    if successor["type"] == "resource":
                        successor["output"] = thread.output

                frontier_vertices += successors_vertices
        return frontier_vertices, thread
