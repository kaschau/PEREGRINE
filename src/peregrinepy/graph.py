"""What a step does, as a fixed flow of slots filled from the solver's kernel
dict, with the boundary conditions at their fixed points. The right-hand
side is a primary flux, then the diffusive flux, then the apply; making a
state consistent is the exchange, the equation of state, the boundary
conditions' euler bcHook -- each leaves its halo's state whole, through the
eos -- and the transport. Each node says
what it reads and writes, and the graph links each to the last writer of
what it reads and to every reader since the last write of what it writes,
so the order is a valid schedule and the links are the input for running
independent nodes at once. The nominal Navier-Stokes path only: shock
handling, subgrid models, sponges and chemistry come back with the
composition round.

A flow is the one place order lives. It makes no kernels and holds no
arrays: every slot is filled from the solver's dict, and a tag the dict
lacks is a bug in the flow, not a null."""

from .backend.abi import lib
from .kernel import BaseKernelGroup, BlockFaceKernel


class BaseNode:
    """One thing a step does: what it reads and writes, by array name, and
    which earlier nodes it waits on."""

    def __init__(self, name, reads=(), writes=()):
        self.name = name
        self.reads, self.writes = tuple(reads), tuple(writes)
        self.deps = []

    def run(self):
        raise NotImplementedError

    def __repr__(self):
        waits = ", ".join(d.name for d in self.deps) or "nothing"
        reads, writes = ", ".join(self.reads) or "-", ", ".join(self.writes) or "-"
        return f"{self.name}: reads {reads}; writes {writes}; after {waits}"


class CellCenterLaunchNode(BaseNode):
    """One kernel of the dict over the cell centers of the solver's blocks,
    with the scalars its slot settles; narrowed to the interior when the
    flow says so."""

    def __init__(self, name, kernel, **fixed):
        super().__init__(name, kernel.reads, kernel.writes)
        self.kernel, self.fixed = kernel, fixed

    def run(self):
        self.kernel(**self.fixed)


class CellFaceLaunchNode(BaseNode):
    """A flux scheme over the cell faces of the solver's blocks, its kernels
    placed by the group's stages: a stage's kernels are siblings in a
    captured graph, hung off one node and joined after all, so the device
    may run them in any order or together; a direct launch has one queue
    and runs them in order either way."""

    def __init__(self, name, group, **fixed):
        super().__init__(name, group.reads, group.writes)
        self.group, self.fixed = group, fixed

    def run(self):
        for stage in self.group.stages:
            lib.pgGraphFork()
            for kernel in stage:
                lib.pgGraphSibling()
                kernel(**self.fixed)
            lib.pgGraphJoin()


class RemoteHalosNode(BaseNode):
    """A cell kernel again, over the halo cells behind the faces whose halos
    arrived by message: the kernel ran over everything while they were in
    flight, and only these were stale. A rank with none launches nothing."""

    def __init__(self, name, kernel, solver, **fixed):
        super().__init__(name, kernel.reads, kernel.writes)
        self.kernel, self.solver, self.fixed = kernel, solver, fixed

    def run(self):
        table, _ = self.solver.remoteFaceTables()
        if table.count:
            self.kernel(table=table, **self.fixed)


class RemoteFacesNode(BaseNode):
    """The flux schemes again, over the planes of faces lying on the block
    faces whose halos arrived by message: the advective flux sets them and
    the viscous flux adds to them, as they were for every face while the
    halos were in flight."""

    def __init__(self, name, groups, solver):
        reads = dict.fromkeys(r for g in groups for r in g.reads)
        writes = dict.fromkeys(w for g in groups for w in g.writes)
        super().__init__(name, reads, writes)
        self.groups, self.solver = groups, solver

    def run(self):
        _, byAxis = self.solver.remoteFaceTables()
        for group in self.groups:
            for kernel in group.kernels:
                if byAxis[kernel.direction].count:
                    kernel(table=byAxis[kernel.direction])


class BlockFaceLaunchNode(BaseNode):
    """A launch over the halo cells of the block faces at one bcHook of the
    flow, a table per bcType: the solver's block face tables for the bcHook,
    or the ones a call gives."""

    def __init__(self, name, solver, bcHook, reads=(), writes=()):
        super().__init__(name, reads, writes)
        self.solver, self.bcHook = solver, bcHook

    def tables(self, given=None):
        return self.solver.blockFaceTables(self.bcHook) if given is None else given

    def run(self, tables=None):
        raise NotImplementedError


class BcNode(BlockFaceLaunchNode):
    """The bcs at one bcHook: every bcType with a kernel
    there, each on the block faces that carry it. It reads and writes what
    any of them does."""

    def __init__(self, solver, bcHook):
        self.kernels = {
            k.bcType: k
            for k in solver.kernels.values()
            if isinstance(k, BlockFaceKernel) and k.bcHook == bcHook
        }
        kernels = self.kernels.values()
        super().__init__(
            f"bcs {bcHook}",
            solver,
            bcHook,
            dict.fromkeys(r for k in kernels for r in k.reads),
            dict.fromkeys(w for k in kernels for w in k.writes),
        )

    def run(self, tables=None):
        for bcType, table in self.tables(tables).items():
            self.kernels[bcType](table)


class HaloExchangeNode(BaseNode):
    """A halo exchange of one array, in three parts a flow lays apart: the
    start packs and copies the messages out beside the kernels, the send
    posts them once the copy has landed, the finish receives and unpacks;
    what the flow puts between them runs while the messages fly. The start
    reads the array; the finish writes its halos. The three are made
    together."""

    def __init__(self, haloExchange, array):
        super().__init__(f"haloExchange {array} start", (array,), ())
        self.haloExchange, self.array = haloExchange, array
        self.pending = None
        self.send = HaloExchangePartNode(self, "send")
        self.finish = HaloExchangePartNode(self, "finish")

    def run(self):
        self.pending = self.haloExchange.start(self.array)


class HaloExchangePartNode(BaseNode):
    def __init__(self, start, part):
        writes = (start.array,) if part == "finish" else ()
        super().__init__(f"haloExchange {start.array} {part}", (), writes)
        self.start, self.part = start, part

    def run(self):
        exchange, array = self.start.haloExchange, self.start.array
        if self.part == "send":
            self.start.pending = exchange.send(array, self.start.pending)
        else:
            exchange.finish(array, self.start.pending)
            self.start.pending = None


class Graph:
    """Nodes in the order the flow lays them, each depending on the last
    writer of what it reads and on every reader since the last write of
    what it writes."""

    def __init__(self, solver, name):
        self.solver, self.name = solver, name
        self.nodes = []
        # the flow as Kokkos graphs, once captured: runs of nodes recorded
        # as one graph each, split where an exchange has to go through the
        # host, in the order they run; a flow run once, on arrays that are
        # then released, is not worth recording. On a device a graph is one
        # submission of what was many launches; the host runs its nodes in
        # order, so the same path is what the suite exercises
        self.segments = None
        self.oneShot = False

    def add(self, node):
        self.nodes.append(node)
        return node

    def slot(self, tag, **fixed):
        """The kernel under :tag: as a node: a cell-face scheme's directions
        as one, anything else a cell-center launch; a tag the solver did not
        define is a bug in the flow."""
        if tag not in self.solver.kernels:
            raise KeyError(f"the {self.name} flow needs {tag}, which is not defined")
        kernel = self.solver.kernels[tag]
        if isinstance(kernel, BaseKernelGroup):
            self.add(CellFaceLaunchNode(tag, kernel, **fixed))
        else:
            self.add(CellCenterLaunchNode(tag, kernel, **fixed))

    def redo(self, *tags, **fixed):
        """The kernels under :tags: again, over what a message brought: a
        cell kernel over the remote faces' halo cells, flux schemes over
        the remote faces' planes, in the order given."""
        kernels = [self._kernel(t) for t in tags]
        if all(isinstance(k, BaseKernelGroup) for k in kernels):
            return self.add(
                RemoteFacesNode(f"{' '.join(tags)} remote", kernels, self.solver)
            )
        (kernel,) = kernels
        return self.add(
            RemoteHalosNode(f"{tags[0]} remote", kernel, self.solver, **fixed)
        )

    def _kernel(self, tag):
        if tag not in self.solver.kernels:
            raise KeyError(f"the {self.name} flow needs {tag}, which is not defined")
        return self.solver.kernels[tag]

    def bcs(self, bcHook):
        """This flow's bc node for a bcHook; None if it has
        none."""
        return next(
            (n for n in self.nodes if isinstance(n, BcNode) and n.bcHook == bcHook),
            None,
        )

    def _link(self):
        lastWriter, readersSince = {}, {}
        for node in self.nodes:
            deps = []
            for array in node.reads:
                if array in lastWriter:
                    deps.append(lastWriter[array])
            for array in node.writes:
                deps += readersSince.get(array, [])
                if array in lastWriter:
                    deps.append(lastWriter[array])
            node.deps = [
                d for i, d in enumerate(deps) if d is not node and d not in deps[:i]
            ]
            for array in node.reads:
                readersSince.setdefault(array, []).append(node)
            for array in node.writes:
                lastWriter[array] = node
                readersSince[array] = []

    def check(self, arrays):
        """Every block array a cell launch or a halo exchange names is one the
        solver defined; the block face launches work on block face tables."""
        for node in self.nodes:
            if not isinstance(node, BlockFaceLaunchNode):
                for name in (*node.reads, *node.writes):
                    if name not in arrays:
                        raise KeyError(
                            f"{self.name}: {node.name} names {name}, which no block holds"
                        )

    def run(self):
        """The flow, as the graphs it was recorded into the first time; a
        one-shot flow node by node."""
        if self.oneShot:
            for node in self.nodes:
                node.run()
            return
        # a face that changed changes the bc tables every flow was recorded with
        if self.solver.facesChanged:
            self.solver.dropGraphs()
        if self.segments is None:
            self._capture()
        for kind, item in self.segments:
            if kind == "graph":
                lib.pgGraphSubmit(item)
            else:
                item.run()

    def _capture(self):
        """Record the flow: every launch becomes a node of the graph under
        capture, and an exchange with a message to pass through the host
        ends one graph and begins the next. The nodes' arguments are fixed
        after binding, which is what a graph replays; a flow whose arrays
        or faces change is dropped and captured again."""
        self.segments = []
        graph = None
        for node in self.nodes:
            remote = (
                node.haloExchange.remote
                if isinstance(node, HaloExchangeNode)
                else (
                    node.start.haloExchange.remote
                    if isinstance(node, HaloExchangePartNode)
                    else False
                )
            )
            if remote:
                if graph is not None:
                    lib.pgGraphEnd()
                    graph = None
                self.segments.append(("node", node))
                continue
            if graph is None:
                graph = lib.pgGraphBegin()
                self.segments.append(("graph", graph))
            node.run()
        if graph is not None:
            lib.pgGraphEnd()

    def drop(self):
        """Forget the captured graphs: the next run records them again."""
        for kind, item in self.segments or ():
            if kind == "graph":
                lib.pgGraphDrop(item)
        self.segments = None

    def __repr__(self):
        return f"{self.name}:\n" + "\n".join(f"  {n!r}" for n in self.nodes)

    ###########################################################################
    # The nominal flows
    ###########################################################################
    @classmethod
    def consistify(cls, solver, fromPrims=False):
        """From the conserved state to everything derived from it, halos and
        boundary conditions included; :fromPrims: makes the conserved state
        first, from the primitive vector a case starts from."""
        g = cls(solver, "consistifyFromPrims" if fromPrims else "consistify")
        if fromPrims:
            g.oneShot = True
            g.slot("stateFromPrims")
        # everything runs while the halos are in flight; only the halos a
        # message brings are stale then, and they are done again after
        viscous = solver.config["RHS"]["diffusion"]
        exchange = g.add(HaloExchangeNode(solver.haloExchange, "Q"))
        g.slot("stateFromCons")
        g.add(exchange.send)
        g.add(BcNode(solver, "euler"))
        if viscous:
            g.slot("trans")
        g.add(exchange.finish)
        g.redo("stateFromCons")
        if viscous:
            g.redo("trans")
        g._link()
        return g

    @classmethod
    def rhs(cls, solver):
        """dQ/dt from a consistent state: the advective and diffusive flux
        differences. Every flux accumulates on the faces, a direction at a
        time, and one apply makes dQ of them."""
        g = cls(solver, "rhs")
        if solver.config["RHS"]["diffusion"]:
            # the gradients go out first and every flux runs while they
            # are in flight; the faces on a remote block face, whose halo
            # gradients were stale, are done again once they have landed
            g.add(BcNode(solver, "preDqDxyz"))
            g.slot("dqdxyz")
            exchange = g.add(HaloExchangeNode(solver.haloExchange, "grads"))
            g.slot("primaryAdvFlux")
            g.add(exchange.send)
            g.add(BcNode(solver, "postDqDxyz"))
            g.slot("diffFlux")
            g.add(exchange.finish)
            g.redo("primaryAdvFlux", "diffFlux")
        else:
            g.slot("primaryAdvFlux")
        g.slot("applyFlux")
        g._link()
        return g
