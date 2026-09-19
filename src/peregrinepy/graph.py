"""What a step does, as graphs: each one device graph of launches, captured
the first time it runs and submitted as one from then on, with the host
steps of the halo exchanges run before it is submitted and after it is. A
stage of the step -- making the state consistent, taking the right-hand
side -- is a list of graphs, cut where a message has to be waited on.

A node is one collective launch: its kernels, and in names what they run
over -- a range, which block faces, which array is exchanged -- bound to
the means a solver holds: its block and block face tables and its
exchanges by array; which block faces a node runs over it reads off the
block-face table's rows. Bound, it is stages of (kernel,
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

    def bind(self, blockTable, blockFaceTable, exchanges):
        """Pairs this node's kernels with the tables and tilings they run
        over, from the means: the block and block face tables, and the
        halo exchanges by array."""
        self.stages = self._stages(blockTable, blockFaceTable, exchanges)
        return self

    def _stages(self, blockTable, blockFaceTable, exchanges):
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

    def _stages(self, blockTable, blockFaceTable, exchanges):
        table = blockTable
        return [
            [(k, table, table.tiling(k, self.rangeName)) for k in stage]
            for stage in self.kernel.stages
        ]


class BCNode(BaseLaunchNode):
    """The boundary conditions at one hook -- a group of bc kernels -- each
    over the halo cells behind the block faces carrying its bcType, as one
    stage since their faces are disjoint. No boundary waits on a message:
    a periodic, the one connected face with something to do to its halo,
    is done by the exchange as the halo lands."""

    def __init__(self, group):
        super().__init__(group.__name__)
        self.group = group

    def _stages(self, blockTable, blockFaceTable, exchanges):
        table = blockFaceTable
        stage = []
        for kernel in self.group.kernels:
            mine = [f for f in table.entries if f.bcType == kernel.bcType]
            key = ("halo", kernel.bcType)
            stage.append((kernel, table, table.tilingOver(key, mine, kernel.behind)))
        return [stage]


class RedoNode(BaseLaunchNode):
    """Kernels or groups again over what a message brought to the block
    faces connected to another rank: each over the faces that concern it
    -- a cell kernel over their halo cells, a flux kernel over the plane
    of cell faces on those of its axis -- in order."""

    def __init__(self, *kernels):
        super().__init__(f"redo {' '.join(k.__name__ for k in kernels)}")
        self.kernels = kernels

    def _stages(self, blockTable, blockFaceTable, exchanges):
        table = blockFaceTable
        faces = [f for f in table.entries if f.connOffRank]
        stages = []
        for kernel in self.kernels:
            for stage in kernel.stages:
                launches = []
                for k in stage:
                    key = ("offRank", k.items, getattr(k, "direction", None))
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

    def _stages(self, blockTable, blockFaceTable, exchanges):
        exchange = exchanges[self.array]
        self.fixed = exchange.packArgs
        return [[(exchange.pack, blockFaceTable, exchange.tilings["pack"])]]


class BaseLandingNode(BaseLaunchNode):
    """A launch that lands one array's halos behind some block faces, then
    as a stage after it turns the vectors of those landed through a
    rotational periodic: what a periodic does to its halo is the
    exchange's. Each stage runs with its own scalars, the exchange's for
    its kernel."""

    def __init__(self, array):
        super().__init__(f"{self.landing} {array}")
        self.array = array

    def _stages(self, blockTable, blockFaceTable, exchanges):
        exchange = exchanges[self.array]
        kernel, tiling, scalars = self._landing(exchange)
        self.scalars = [scalars]
        stages = [[(kernel, blockFaceTable, tiling)]]
        if exchange.vectors is not None:
            self.scalars.append(exchange.turnArgs)
            turn = exchange.tilings[self.turning]
            stages.append([(exchange.turnHalo, blockFaceTable, turn)])
        return stages

    def run(self, **given):
        for stage, scalars in zip(self.stages, self.scalars):
            lib.pgGraphFork()
            for kernel, table, tiling in stage:
                lib.pgGraphSibling()
                kernel(table, tiling, **{**scalars, **given})
            lib.pgGraphJoin()


class UnpackNode(BaseLandingNode):
    """The unpack of one array's halos behind the block faces connected to
    another rank, from what their messages brought."""

    landing, turning = "unpack", "turnUnpacked"

    def _landing(self, exchange):
        return exchange.unpack, exchange.tilings["unpack"], exchange.unpackArgs


class DirectHaloFillNode(BaseLandingNode):
    """The direct fill of one array's halos behind the block faces
    connected on this rank, from the blocks across."""

    landing, turning = "directHaloFill", "turnFilled"

    def _landing(self, exchange):
        return (
            exchange.directHaloFill,
            exchange.tilings["directHaloFill"],
            exchange.directHaloFillArgs,
        )


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
    nodes :ahead: of it, the pack and the direct fill of the halos met here;
    whatever runs :during: the messages' flight; then the unpack and
    whatever comes :after:, redoing the halos they brought -- the
    exchange's host steps around each."""

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
                [*self.ahead, PackNode(name), DirectHaloFillNode(name)],
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
                [UnpackNode(name), *self.after],
                after=[ex.sent],
            ),
        ]
        for g in graphs:
            g.bind(*means)
        return graphs
