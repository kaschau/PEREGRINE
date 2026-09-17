"""What a step does, as graphs: each one device graph of launches, captured
the first time it runs and submitted as one from then on, with the host
steps of the halo exchanges run before it is submitted and after it is. A
stage of the step -- making the state consistent, taking the right-hand
side -- is a list of graphs, cut where a message has to be waited on. The
solver builds them in declareGraphs from its kernels, tables, tilings and
exchanges; a graph holds only its nodes and steps, never the solver.

A node is one collective launch: kernels over tables and tilings, in
stages of kernels that read and write nothing of each other's, so a
captured graph may run a stage's kernels in any order or together."""

from .backend.abi import lib


class CollectiveLaunchNode:
    """Kernels over the tables and tilings a graph names, with the scalars
    it settles: stages of (kernel, table, tiling), each stage after the
    last. Under capture a stage's kernels hang off one node and are
    joined after all, so the device may run them in any order or
    together; a direct launch has one queue and runs them in order."""

    def __init__(self, name, stages, **fixed):
        self.name, self.stages, self.fixed = name, stages, fixed

    def run(self):
        """Launches every stage's kernels, a stage's as siblings."""
        for stage in self.stages:
            lib.pgGraphFork()
            for kernel, table, tiling in stage:
                lib.pgGraphSibling()
                kernel(table, tiling, **self.fixed)
            lib.pgGraphJoin()

    def __repr__(self):
        return self.name


class Graph:
    """One device graph: its launches in order, the host steps run before
    it is submitted and after, and the Kokkos graph they were captured
    into. A step is a bound method of a halo exchange."""

    def __init__(self, name, before=(), after=()):
        self.name = name
        self.nodes = []
        self.before, self.after = list(before), list(after)
        # the device graph, once captured
        self.captured = None

    def add(self, node):
        self.nodes.append(node)
        return node

    def node(self, name):
        """Finds this graph's node of a name; None if it has none."""
        return next((n for n in self.nodes if n.name == name), None)

    def run(self):
        """Runs the steps before, submits the device graph -- capturing it
        the first time -- and runs the steps after."""
        for step in self.before:
            step()
        if self.captured is None:
            self._capture()
        lib.pgGraphSubmit(self.captured)
        for step in self.after:
            step()

    def _capture(self):
        """Captures every node's launches into one device graph."""
        self.captured = lib.pgGraphBegin()
        for node in self.nodes:
            node.run()
        lib.pgGraphEnd()

    def drop(self):
        """Forgets the captured device graph, so the next run captures it
        again: what an array changing identity under it needs."""
        if self.captured is not None:
            lib.pgGraphDrop(self.captured)
            self.captured = None

    def __repr__(self):
        steps = lambda fs: ", ".join(f"{f.__self__.name}.{f.__name__}" for f in fs)
        lines = [f"{self.name}:"]
        if self.before:
            lines.append(f"  before: {steps(self.before)}")
        lines += [f"  {n!r}" for n in self.nodes]
        if self.after:
            lines.append(f"  after: {steps(self.after)}")
        return "\n".join(lines)
