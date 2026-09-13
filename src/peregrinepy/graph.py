"""What a step does and what each part waits on, built once by a solver from
its config and run by it. The nominal Navier-Stokes path only: shock handling,
subgrid models, sponges and chemistry come back as hooks once this is right."""

from .bcs import conditionsOf, getBc
from .files.configFile import pgConfigError
from .kernel import BoundKernel
from .table import Table


class BaseNode:
    """One thing a step does: what it reads and writes, by array name, and
    which earlier nodes it waits on."""

    def __init__(self, name, reads=(), writes=()):
        self.name = name
        self.reads, self.writes = tuple(reads), tuple(writes)
        self.deps = []

    def bind(self, table, thtrdat, communicator):
        """What a node runs with: the case's block table, species data and
        halo exchange; each takes what it needs."""

    def run(self, tme):
        raise NotImplementedError

    def __repr__(self):
        waits = ", ".join(d.name for d in self.deps) or "nothing"
        reads, writes = ", ".join(self.reads) or "-", ", ".join(self.writes) or "-"
        return f"{self.name}: reads {reads}; writes {writes}; after {waits}"


class KernelNode(BaseNode):
    """A kernel over the solver's blocks. Its reads and writes are its
    prototype's; :role: is the name the solver knows it by."""

    def __init__(self, source, role=None, **fixed):
        stem = source.rsplit("/", 1)[-1].removesuffix(".cpp")
        super().__init__(role or stem)
        self.source, self.role, self.fixed = source, role, fixed
        self.kernel = None

    def bind(self, table, thtrdat, communicator):
        self.kernel = BoundKernel(
            table, thtrdat, self.source, role=self.role, **self.fixed
        )
        self.reads, self.writes = tuple(self.kernel.reads), tuple(self.kernel.writes)

    def run(self, tme):
        self.kernel()


class Hook(BaseNode):
    """A boundary condition hook: every face whose condition has it, in one
    call. Every condition with the hook is compiled into the kernel and each
    face's kind picks its own, so the kernel is fixed from the start and only
    the faces are found on the first run, once they know their conditions. It
    reads what any condition may and writes the halos of the state it sets."""

    writesOf = {
        "euler": ("q",),
        "postEos": ("q", "Q"),
        "preDqDxyz": ("q",),
        "postDqDxyz": ("grads",),
    }

    def __init__(self, hook):
        super().__init__(
            f"{hook} hook",
            reads=("q", "Q", "qh", "grads", "S", "qBcVals", "QBcVals"),
            writes=self.writesOf[hook],
        )
        self.hook = hook
        self.kernel = None
        self.faces = []
        self.table = None

    def bind(self, table, thtrdat, communicator):
        conditions = conditionsOf(self.hook)
        self.kernel = BoundKernel(
            table,
            thtrdat,
            f"boundaryConditions/hooks/{self.hook}.cpp",
            tableOf=lambda: self.table,
            role=f"{self.hook}Hook",
            defines=("PG_BCS=" + " ".join(f"X({t})" for t in conditions),),
            includes=tuple(getBc(t).header(self.hook) for t in conditions),
        )

    def connect(self, faces):
        """Which of the case's faces this hook runs over, and their table;
        none, no table."""
        self.faces = [f for f in faces if self.hook in f.bc.hooks]
        self.table = Table.ofFaces(self.faces, self.hook) if self.faces else None

    def run(self, tme, table=None):
        if table is None:
            table = self.table
        if table is not None:
            self.kernel(table, tme=tme)


class FaceState(KernelNode):
    """The state on a hook's faces, once their halos are set."""

    def __init__(self, source, hook, role):
        super().__init__(source, role=role)
        self.name, self.hook = f"{role}@{hook.hook}", hook

    def run(self, tme, table=None):
        if table is None:
            table = self.hook.table
        if table is not None:
            self.kernel(table=table)


class Exchange(BaseNode):
    """A halo exchange of the named arrays: every block's are read, every
    block's halos written."""

    def __init__(self, *arrays):
        super().__init__(f"exchange {' '.join(arrays)}", reads=arrays, writes=arrays)
        self.arrays = list(arrays)
        self.communicator = None

    def bind(self, table, thtrdat, communicator):
        self.communicator = communicator

    def run(self, tme):
        self.communicator.exchange(self.arrays)


class Graph:
    """Nodes in the order they were written; each depends on the last writer
    of what it reads and on every reader since the last write of what it
    writes, so the order is a valid schedule and the dependencies are the
    input for running independent nodes at once."""

    def __init__(self, name):
        self.name = name
        self.nodes = []

    def add(self, node):
        self.nodes.append(node)
        return node

    @property
    def kernels(self):
        """Every kernel this graph's nodes call."""
        return [n.kernel for n in self.nodes if getattr(n, "kernel", None)]

    def hook(self, hook):
        """This graph's node for a boundary condition hook."""
        return next(n for n in self.nodes if isinstance(n, Hook) and n.hook == hook)

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

    def bind(self, table, thtrdat, communicator):
        """Every node's kernel, and the dependencies from the kernels'
        prototypes."""
        for node in self.nodes:
            node.bind(table, thtrdat, communicator)
        self._link()
        return self

    def connect(self, faces):
        """The faces each hook runs over."""
        for node in self.nodes:
            if isinstance(node, Hook):
                node.connect(faces)

    def run(self, tme):
        for node in self.nodes:
            node.run(tme)

    def __repr__(self):
        return f"{self.name}:\n" + "\n".join(f"  {n!r}" for n in self.nodes)

    ###########################################################################
    # The nominal step, from the config
    ###########################################################################
    @staticmethod
    def _nominal(config):
        """What the graph does not describe yet is refused, not run."""
        rhs, mc = config["RHS"], config["mcPhysics"]
        for key in ("shockHandling", "secondaryAdvFlux", "switchAdvFlux", "subgrid"):
            if rhs[key] is not None:
                raise pgConfigError(key, rhs[key], "not in the step graph yet")
        if config["viscousSponge"]["spongeON"]:
            raise pgConfigError("viscousSponge", True, "not in the step graph yet")
        if mc["chemistry"]:
            raise pgConfigError("chemistry", True, "not in the step graph yet")
        if rhs["primaryAdvFlux"] is None:
            raise pgConfigError("primaryAdvFlux", None, "primaryAdvFlux cannot be none")
        ti = config["timeIntegration"]
        if ti["integrator"] == "dualTime":
            if mc["eos"] not in ("cpg", "tpg"):
                raise pgConfigError(
                    "dualTime", mc["eos"], "only cpg and tpg are supported"
                )
            if ti["controller"] != "fixed":
                raise pgConfigError(
                    "dualTime", ti["controller"], "only a fixed time step is supported"
                )

    @classmethod
    def consistify(cls, config, mixture, fromPrims=False):
        """From one state to everything derived from it, halos and boundary
        conditions included."""
        cls._nominal(config)
        eos = config["mcPhysics"]["eos"]
        if eos not in ("cpg", "tpg", "realGas"):
            raise pgConfigError("eos", eos)
        fromPrimsSource = f"thermo/{eos}FromPrims.cpp"
        g = cls("consistifyFromPrims" if fromPrims else "consistify")
        if fromPrims:
            g.add(Exchange("q"))
            g.add(KernelNode(fromPrimsSource, role="stateFromPrims", nface=-1))
        else:
            g.add(Exchange("Q"))
            g.add(
                KernelNode(f"thermo/{eos}FromCons.cpp", role="stateFromCons", nface=-1)
            )
        euler = g.add(Hook("euler"))
        g.add(FaceState(fromPrimsSource, euler, role="stateFromPrims"))
        postEos = g.add(Hook("postEos"))
        g.add(FaceState(fromPrimsSource, postEos, role="stateFromPrims"))
        if config["RHS"]["diffusion"]:
            g.add(
                KernelNode(
                    f"transport/{mixture.transportKernel}.cpp",
                    role="trans",
                    nface=-1,
                )
            )
        return g

    @classmethod
    def rhs(cls, config):
        """dQ/dt from a consistent state: the advective and diffusive flux
        differences."""
        cls._nominal(config)
        rhs = config["RHS"]
        g = cls("rhs")
        g.add(KernelNode("utils/dQzero.cpp"))
        g.add(KernelNode(f"advFlux/{rhs['primaryAdvFlux']}.cpp", role="primaryAdvFlux"))
        g.add(KernelNode("utils/applyFlux.cpp"))
        if rhs["diffusion"]:
            g.add(Hook("preDqDxyz"))
            g.add(KernelNode("utils/dq2FD.cpp", role="dqdxyz"))
            g.add(Exchange("grads"))
            g.add(Hook("postDqDxyz"))
            g.add(KernelNode("diffFlux/alphaDampingFlux.cpp", role="diffFlux"))
            g.add(KernelNode("utils/applyFlux.cpp"))
        return g
