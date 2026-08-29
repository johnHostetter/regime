"""
Implements functions and classes responsible for the design of soft computing solutions,
such as the Regime class.
"""

from typing import Any, Dict, Iterable, List, OrderedDict, Set, Tuple, Union

import igraph
from rough.granulation import RoughGranulation

from regime.flow.components import Process, Resource
from regime.flow.threads import ComponentThread
from regime.nodes import Node
from regime.utils import merge_dicts

# a vertex's "output" attribute starts out as this sentinel rather than None,
# so a process/resource that legitimately produces/holds a value of None can
# be told apart from one that has not produced anything yet (None itself is a
# valid output - see unprepared_resource_vertices, start(), and
# process_frontier()'s predecessor-readiness check, all of which compare
# against this sentinel instead of None for exactly that reason).
_UNSET = object()


class Regime(RoughGranulation):  # graph/tagging plumbing only - see docstring below
    """
    The Regime class facilitates the convenient design of self-organizing solutions
    by adding callable function references to a RoughGranulation graph as vertices, and
    using the edges that connect the vertices (i.e., functions) to determine the flow of
    data/information from function to function.

    Rough-theory base class (2026-08-29): inherits only RoughGranulation (the graph/
    attribute_table plumbing this class actually uses via self.graph throughout),
    not the fuller RoughApproximation -> RoughOperations -> RoughDecisions analysis
    chain it used to inherit via `RoughDecisions` ("inherit from RoughDecisions to
    have access to functionality", the old comment here). Confirmed (grep) this class
    never calls a single RoughApproximation/RoughOperations/RoughDecisions method -
    same finding as fuzzy-theory's KnowledgeBase, a separate, independent inheritor of
    the same chain that got the identical fix in the same session. See
    external/pypi/rough-theory's rough/granulation.py module docstring for the full
    rationale.

    Rough-set analysis (reducts, decision tables) is still available on demand if this
    class's own graph is ever worth analyzing that way:

        from rough.decisions import RoughDecisions
        analysis = RoughDecisions(graph=regime.graph, attribute_table=regime.attribute_table)
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
        # maps a vertex's "name" attribute to its current index in self.graph,
        # so add_resources()/get_vertex() don't need an O(V) igraph scan per
        # call - see _resolve_name_vertex() for how this stays correct across
        # vertex deletions (define_flow()'s clean_up re-numbers every vertex).
        self._name_index: Dict[str, int] = {}

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
        # it must be populated with values from a provided configuration
        # settings in the follow-up

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
            v
            for v in self.graph.vs
            if v["type"] == "resource" and v["output"] is _UNSET
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
            # the output attribute will later be changed once the
            # ComponentThread executes
            if callable(_callable):
                # e.g., name =
                # "fuzzy_ml.clustering.empirical.find_empirical_fuzzy_sets"
                if isinstance(_callable, Node):
                    name: str = f"{_callable.__module__}.{type(_callable).__name__}"
                    # find the hyperparameters required to be specified in
                    # configuration settings
                    required_hyperparameters = merge_dicts(
                        existing=required_hyperparameters,
                        new=_callable.make_hyperparameters_dict(),
                    )
                else:  # if not a Node OBJECT, then it must be a function
                    name: str = f"{_callable.__module__}.{_callable.__name__}"

                processes.append(
                    Process(
                        name=name,
                        callable=_callable,
                        thread=None,
                        output=_UNSET,
                    )
                )
            else:
                raise ValueError(
                    f"Callable {_callable} is not a function or class of Node."
                )
        return processes, required_hyperparameters

    def add_processes(self) -> None:
        """
        Add the processes to the Regime's graph as isolated vertices.

        Returns:
            None
        """
        start_index = self.graph.vcount()
        self.graph.add_vertices(len(self.processes))
        self.graph.vs["type"] = "process"
        self.graph.vs["name"] = [process.name for process in self.processes]
        self.graph.vs["callable"] = [process.callable for process in self.processes]
        self.graph.vs["output"] = [process.output for process in self.processes]
        self.graph.vs["thread"] = [process.thread for process in self.processes]
        for offset, process in enumerate(self.processes):
            self._register_name(process.name, start_index + offset)

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
            if resource in self.resources or self._resolve_name_vertex(resource.name):
                raise ValueError(f"Resource {resource.name} already exists.")
            self.resources.add(resource)
            self.graph.add_vertex(
                type="resource",
                name=resource.name,
                function=None,
                output=resource.value,
                thread=None,
            )
            self._register_name(resource.name, self.graph.vcount() - 1)

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

        # find all None values in remaining_hyperparameters and assign them
        # from the config
        keys = remaining_hyperparameters.keys()
        for key in keys:
            if remaining_hyperparameters[key] is None:
                remaining_hyperparameters[key] = configuration[key]
                # add the hyperparameter (with its value) as a resource for the
                # Regime
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
            # edges are (source, target, arg_order)
            None,
            List[Tuple[Any, Any, int]],
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
        # why is everything optional? --> We can set up the Regime graph in
        # stages

        # define the hyperparameters' values for the processes
        if configuration is not None:
            self.define_hyperparameters(configuration=configuration)

        # add the resources to the graph (check for name conflicts w/
        # hyperparameters)
        if resources is not None:
            self.add_resources(resources=resources)

        if edges is not None:
            self.define_flow(edges=edges, clean_up=clean_up)

    def define_flow(
        self, edges: List[Tuple[Any, Any, Union[int, str]]], clean_up: bool = True
    ) -> None:
        """
        Links callables in the Regime's graph (via RoughGranulation) if they exist as vertices.

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
                    f"An edge already exists between {source_vertex['name']} "
                    f"and {target_vertex['name']}."
                )
            self.graph.add_edge(source_vertex, target_vertex, arg=arg_specifier)
            if self.verbose:
                print("Done.")

        # clean up the graph by removing any isolated vertices
        if clean_up:
            self.graph.delete_vertices(
                [v.index for v in self.graph.vs if v.degree() == 0]
            )

    def _register_name(self, name: str, index: int) -> None:
        """
        Record a vertex's current index under its "name" attribute, so later
        lookups by name (add_resources()/get_vertex()) can skip the O(V)
        igraph scan. Names are expected to be unique - add_resources() already
        enforces this before a vertex is ever added.

        Returns:
            None
        """
        self._name_index[name] = index

    def _rebuild_name_index(self) -> None:
        """
        Recompute the entire name -> index cache from the live graph. Needed
        after any operation that re-numbers vertices (define_flow()'s
        clean_up deletes isolated vertices, which shifts every later index).

        Returns:
            None
        """
        self._name_index = {v["name"]: v.index for v in self.graph.vs}

    def _resolve_name_vertex(self, name: str) -> Union[None, igraph.Vertex]:
        """
        Look up a vertex by its "name" attribute in O(1) (amortized), falling
        back to a full rebuild-and-retry if the cached index is stale (e.g.
        vertices were deleted, shifting indices) - the same verify-then-
        rebuild-once strategy used by rough-theory's granulation.py
        _item_index, for the same reason: trusting a cached index without
        checking it first risks silently returning the wrong vertex instead
        of the intended one.

        Returns:
            The matching vertex, or None if no vertex has this name.
        """
        index = self._name_index.get(name)
        vertex_count = self.graph.vcount()
        if (
            index is not None
            and index < vertex_count
            and self.graph.vs[index]["name"] == name
        ):
            return self.graph.vs[index]
        self._rebuild_name_index()
        index = self._name_index.get(name)
        return None if index is None else self.graph.vs[index]

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
            vertex: Union[None, igraph.Vertex] = self._resolve_name_vertex(
                vertex_reference
            )
            if vertex is None:
                # this is a new resource that does not yet exist
                # go ahead and make it - it may or may not be used later
                new_resource = Resource(
                    name=vertex_reference, value=_UNSET
                )  # no value yet
                self.add_resources(resources={new_resource})
                vertex = self._resolve_name_vertex(vertex_reference)
        else:  # find the vertex by callable, most likely process
            try:
                vertex: igraph.Vertex = self.graph.vs.find(callable_eq=vertex_reference)
            except ValueError as e:
                raise ValueError(
                    f"Vertex {vertex_reference} does not exist in the Regime graph."
                ) from e
        return vertex

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
        # incoming edges (i.e., no process predecessors - they wait for nothing
        # to finish)
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

        # note: functions must be unique or the below won't work (e.g., no 2
        # calls to 'train')
        while True:
            if len(frontier_vertices) == 0:
                break
            frontier_vertex = frontier_vertices.pop()
            if frontier_vertex["type"] == "resource":
                if frontier_vertex["output"] is not _UNSET:
                    # if the resource has output, then it is already complete
                    # expand the frontier to include the successors of the
                    # resource
                    frontier_vertices += frontier_vertex.successors()
                    continue  # skip the rest of the loop
                raise ValueError(
                    "Somehow a resource has been reached with no output (check Regime's flow)."
                )
            if frontier_vertex["type"] == "process":
                frontier_vertices, thread = self.process_frontier(
                    frontier_vertex, frontier_vertices, thread
                )

        # every connected process must have executed by now - the only way one
        # wouldn't have is a cycle among processes (each waiting on another's
        # output, so none of them ever had every predecessor ready and none is
        # an entry point with indegree 0 either) - the isolated-vertex
        # discovery above silently never queues such a cycle at all, so
        # without this check it would otherwise be dropped with no error.
        incomplete_processes: List[igraph.Vertex] = [
            v
            for v in self.process_vertices
            if v.degree() > 0 and v not in self.complete_process_vertices
        ]
        if incomplete_processes:
            names = ", ".join(v["name"] for v in incomplete_processes)
            raise ValueError(
                f"The following process(es) never executed even though they "
                f"are connected in the Regime's graph: {names}. This usually "
                "means the graph contains a cycle among these processes "
                "(each waiting on another's output), so none of them ever "
                "had every predecessor's output ready."
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
        # a vertex with 2+ predecessors is appended to the frontier once per
        # predecessor that completes (see the successors_vertices expansion below) -
        # if two or more of those completions happen before any of the resulting
        # frontier entries for this vertex is popped, every one of those entries
        # will independently see all predecessors ready and re-execute `function`.
        # complete_process_vertices is populated below specifically to prevent
        # this; this guard is what actually enforces it.
        if (
            frontier_vertex.degree() > 0  # only consider connected functions
            and frontier_vertex not in self.complete_process_vertices
        ):
            if len(predecessors_vertices) == 0 or all(  # no predecessors are fine or
                source_vertex["output"]
                is not _UNSET  # for all predecessors, we have output
                for source_vertex in predecessors_vertices
            ):
                # get the keyword arguments for this function
                kwargs: Dict[str, Any] = {
                    v["name"]: v["output"] for v in predecessors_vertices
                }
                # if isinstance(function, Node):
                #     function = function()
                thread: ComponentThread = ComponentThread(function, **kwargs)
                # keep a reference to the thread
                frontier_vertex["thread"] = thread
                thread.output = thread.function(**kwargs)
                # thread.start()
                # # thread.join()
                # retrieve thread output - frontier_vertex IS the vertex this
                # thread was just assigned to two lines above, so no lookup is
                # needed (a `self.graph.vs.find(thread_eq=thread)` re-scan
                # used to be done here instead, an unnecessary O(V) cost paid
                # on every single process execution)
                frontier_vertex["output"] = thread.output
                self.complete_process_vertices.add(frontier_vertex)
                # update any resources that have been modified by the function
                successors_vertices = frontier_vertex.successors()
                for successor in successors_vertices:
                    if successor["type"] == "resource":
                        successor["output"] = thread.output

                frontier_vertices += successors_vertices
        return frontier_vertices, thread
