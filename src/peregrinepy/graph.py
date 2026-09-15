"""What a step does, as a fixed flow of slots filled from the solver's kernel
dict, with the boundary conditions at their fixed points. The right-hand
side is a primary flux, then the diffusive flux, then the apply; making a
state consistent is the exchange, the equation of state, the boundary
conditions' euler bcHook, the faces' state, and the transport. Each node says
what it reads and writes, and the graph links each to the last writer of
what it reads and to every reader since the last write of what it writes,
so the order is a valid schedule and the links are the input for running
independent nodes at once. The nominal Navier-Stokes path only: shock
handling, subgrid models, sponges and chemistry come back with the
composition round.

A flow is the one place order lives. It makes no kernels and holds no
arrays: every slot is filled from the solver's dict, and a tag the dict
lacks is a bug in the flow, not a null."""

from .kernel import BlockFaceKernel, KernelGroup


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
    with the scalars its slot settles."""

    def __init__(self, name, kernel, **fixed):
        super().__init__(name, kernel.reads, kernel.writes)
        self.kernel, self.fixed = kernel, fixed

    def run(self):
        self.kernel(**self.fixed)


class CellFaceLaunchNode(BaseNode):
    """A flux scheme over the cell faces of the solver's blocks: its three
    directions' kernels, run one after the other."""

    def __init__(self, name, group):
        super().__init__(name, group.reads, group.writes)
        self.group = group

    def run(self):
        self.group()


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


class BlockFaceStateNode(BlockFaceLaunchNode):
    """One cell-center kernel on every block face table of a bcHook: the
    state in the halo cells, once the bcs have set them."""

    def __init__(self, solver, bcHook, kernel):
        super().__init__(
            f"stateFromPrims@{bcHook}", solver, bcHook, kernel.reads, kernel.writes
        )
        self.kernel = kernel

    def run(self, tables=None):
        for table in self.tables(tables).values():
            self.kernel(table=table)


class HaloExchangeNode(BaseNode):
    """A halo exchange of the named arrays: every block's are read, every
    block's halos written."""

    def __init__(self, haloExchange, *arrays):
        super().__init__(f"haloExchange {' '.join(arrays)}", arrays, arrays)
        self.haloExchange, self.arrays = haloExchange, list(arrays)

    def run(self):
        self.haloExchange.exchange(self.arrays)


class Graph:
    """Nodes in the order the flow lays them, each depending on the last
    writer of what it reads and on every reader since the last write of
    what it writes."""

    def __init__(self, solver, name):
        self.solver, self.name = solver, name
        self.nodes = []

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
        if isinstance(kernel, KernelGroup):
            self.add(CellFaceLaunchNode(tag, kernel))
        else:
            self.add(CellCenterLaunchNode(tag, kernel, **fixed))

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
        for node in self.nodes:
            node.run()

    def __repr__(self):
        return f"{self.name}:\n" + "\n".join(f"  {n!r}" for n in self.nodes)

    ###########################################################################
    # The nominal flows
    ###########################################################################
    @classmethod
    def consistify(cls, solver, fromPrims=False):
        """From one state to everything derived from it, halos and boundary
        conditions included."""
        g = cls(solver, "consistifyFromPrims" if fromPrims else "consistify")
        g.add(HaloExchangeNode(solver.haloExchange, "q" if fromPrims else "Q"))
        g.slot("stateFromPrims" if fromPrims else "stateFromCons")
        g.add(BcNode(solver, "euler"))
        g.faceState = g.add(
            BlockFaceStateNode(solver, "euler", solver.kernels["stateFromPrims"])
        )
        if solver.config["RHS"]["diffusion"]:
            g.slot("trans")
        g._link()
        return g

    @classmethod
    def rhs(cls, solver):
        """dQ/dt from a consistent state: the advective and diffusive flux
        differences. Every flux accumulates on the faces, a direction at a
        time, and one apply makes dQ of them."""
        g = cls(solver, "rhs")
        g.slot("primaryAdvFlux")
        if solver.config["RHS"]["diffusion"]:
            g.add(BcNode(solver, "preDqDxyz"))
            g.slot("dqdxyz")
            g.add(HaloExchangeNode(solver.haloExchange, "grads"))
            g.add(BcNode(solver, "postDqDxyz"))
            g.slot("diffFlux")
        g.slot("applyFlux")
        g._link()
        return g
