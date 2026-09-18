"""What a step does, as graphs: each one device graph of launches, captured
the first time it runs and submitted as one from then on, with the host
steps of the halo exchanges run before it is submitted and after it is. A
stage of the step -- making the state consistent, taking the right-hand
side -- is a list of graphs, cut where a message has to be waited on.

A node is one collective launch: its kernels, and in names what they run
over -- a range, which block faces, which array is exchanged -- bound to
the means a solver holds: its block and block face tables, its block
faces by kind, its exchanges by array. Bound, it is stages of (kernel,
table, tiling): a stage's kernels read and write nothing of each other's,
which the launch says with a fork and a join, whether or not the runtime
runs them together (today it does not: see pgGraphFork). A physics or an
integrator says its graphs from these; a graph holds only its nodes and
steps."""

from .backend.abi import lib


class BaseLaunchNode:
    """One collective launch: its stages, once bound, of (kernel, table,
    tiling), each stage after the last, and the scalars it settles. A
    stage's kernels are forked and joined, siblings that owe each other no
    order; the runtime may run them together or in order."""

    def __init__(self, name, **fixed):
        self.name, self.fixed = name, fixed
        self.stages = None

    def bind(self, blockTable, blockFaceTable, blockFacesBy, exchanges):
        """Pairs this node's kernels with the tables and tilings they run
        over, from the means: the block and block face tables, the block
        faces by kind (all, here, remote), and the halo exchanges by
        array."""
        self.stages = self._stages(blockTable, blockFaceTable, blockFacesBy, exchanges)
        return self

    def _stages(self, blockTable, blockFaceTable, blockFacesBy, exchanges):
        raise NotImplementedError

    def run(self, **given):
        """Launches every stage's kernels, a stage's as siblings; :given:
        supplies scalars for this run over the fixed ones."""
        scalars = {**self.fixed, **given}
        for stage in self.stages:
            lib.pgGraphFork()
            for kernel, table, tiling in stage:
                lib.pgGraphSibling()
                kernel(table, tiling, **scalars)
            lib.pgGraphJoin()

    def __repr__(self):
        return self.name


class LaunchNode(BaseLaunchNode):
    """A kernel or group over the named range of every block."""

    def __init__(self, kernel, rangeName, **fixed):
        super().__init__(kernel.__name__, **fixed)
        self.kernel, self.rangeName = kernel, rangeName

    def _stages(self, blockTable, blockFaceTable, blockFacesBy, exchanges):
        table = blockTable
        return [
            [(k, table, table.tiling(k, self.rangeName)) for k in stage]
            for stage in self.kernel.stages
        ]


class BCNode(BaseLaunchNode):
    """The boundary conditions at one hook -- a group of bc kernels -- over
    all the block faces, the ones whose halos are here, or the ones met on
    another rank: each kernel over the halo cells behind the faces
    carrying its bcType, as one stage since their faces are disjoint."""

    def __init__(self, group, where="all"):
        super().__init__(f"{group.__name__} {where}")
        self.group, self.where = group, where

    def _stages(self, blockTable, blockFaceTable, blockFacesBy, exchanges):
        table, faces = blockFaceTable, blockFacesBy[self.where]
        stage = []
        for kernel in self.group.kernels:
            mine = [f for f in faces if f.bcType == kernel.bcType]
            key = ("halo", kernel.bcType, self.where)
            stage.append((kernel, table, table.tilingOver(key, mine, kernel.behind)))
        return [stage]


class RedoNode(BaseLaunchNode):
    """Kernels or groups again over what a message brought to the block
    faces met on another rank: each over the faces that concern it -- a
    cell kernel over their halo cells, a flux kernel over the plane of cell
    faces on those of its axis -- in order."""

    def __init__(self, *kernels):
        super().__init__(f"redo {' '.join(k.__name__ for k in kernels)}")
        self.kernels = kernels

    def _stages(self, blockTable, blockFaceTable, blockFacesBy, exchanges):
        table, faces = blockFaceTable, blockFacesBy["remote"]
        stages = []
        for kernel in self.kernels:
            for stage in kernel.stages:
                launches = []
                for k in stage:
                    key = ("remote", k.items, getattr(k, "direction", None))
                    mine = [f for f in faces if k.concerns(f)]
                    launches.append((k, table, table.tilingOver(key, mine, k.behind)))
                stages.append(launches)
        return stages


class PackNode(BaseLaunchNode):
    """The pack of one array's halos for their neighbors, over every
    trading block face."""

    def __init__(self, array):
        super().__init__(f"pack {array}")
        self.array = array

    def _stages(self, blockTable, blockFaceTable, blockFacesBy, exchanges):
        exchange = exchanges[self.array]
        self.fixed = exchange.packArgs
        return [[(exchange.pack, blockFaceTable, exchange.tilings["pack"])]]


class UnpackNode(BaseLaunchNode):
    """The unpack of one array's halos behind the block faces met on this
    rank (local) or on another (remote)."""

    def __init__(self, array, which):
        super().__init__(f"unpack {array} {which}")
        self.array, self.which = array, which

    def _stages(self, blockTable, blockFaceTable, blockFacesBy, exchanges):
        exchange = exchanges[self.array]
        self.fixed = exchange.unpackArgs
        return [[(exchange.unpack, blockFaceTable, exchange.tilings[self.which])]]


class Graph:
    """One device graph: its nodes in order, the host steps run before it
    is submitted and after, and the Kokkos graph they were captured into.
    A step is a bound method of a halo exchange."""

    def __init__(self, name, nodes=(), before=(), after=()):
        self.name = name
        self.nodes = list(nodes)
        self.before, self.after = list(before), list(after)
        # the device graph, once captured
        self.captured = None

    def bind(self, *means):
        """Binds every node to the means; a list, as the exchange graphs
        give."""
        for node in self.nodes:
            node.bind(*means)
        return [self]

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


class ExchangeGraphs:
    """The three graphs a halo exchange of one array is cut into: the
    nodes :ahead: of it, the pack and the local unpack; whatever runs
    :during: the messages' flight; then the remote unpack and whatever
    comes :after:, redoing the halos they brought -- the exchange's host
    steps around each."""

    def __init__(self, name, array, ahead=(), during=(), after=()):
        self.name, self.array = name, array
        self.ahead, self.during, self.after = ahead, during, after

    def bind(self, *means):
        """Makes the three graphs, bound to the means' exchange of the
        array."""
        exchanges = means[-1]
        ex, name, prefix = exchanges[self.array], self.array, self.name
        graphs = [
            Graph(
                f"{prefix}: pack {name}",
                [*self.ahead, PackNode(name), UnpackNode(name, "local")],
                before=[ex.expect],
                after=[ex.copyOut],
            ),
            Graph(
                f"{prefix}: while {name} flies",
                self.during,
                after=[ex.send, ex.receive],
            ),
            Graph(
                f"{prefix}: unpack {name}",
                [UnpackNode(name, "remote"), *self.after],
                after=[ex.sent],
            ),
        ]
        for g in graphs:
            g.bind(*means)
        return graphs
