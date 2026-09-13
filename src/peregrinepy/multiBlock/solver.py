import numpy as np
from mpi4py import MPI

from .restart import restart
from .solverBlock import solverBlock
from ..graph import Graph
from ..integrators import getController, getIntegrator
from ..jit import Jit
from ..kernel import BoundKernel
from ..mixture import Mixture
from ..mpiComm import Communicator
from ..mpiComm.mpiUtils import getCommRankSize
from ..plugins import pluginsOf
from ..table import Table
from ..files.configFile import pgConfigError
from .thtrdat import thtrdat


class solver(restart):
    """A runnable case: the blocks, the mixture and its species data, the
    step's graphs, the boundary conditions, the time integrator, and every
    kernel the config asks for, compiled and bound."""

    hasConservatives = True

    def _newBlock(self, nblki):
        return solverBlock(
            nblki, self.speciesNames, self.ng, self.config, self, len(self.blocks)
        )

    def __init__(self, config, mesh, state=None):
        """A case, ready to step. The step's graphs and every kernel they call
        are compiled for the config; the blocks come from :mesh: -- a mesher
        or a grid reader, anything that fills a multiBlock -- and are
        connected, haloed and metricked, with the boundary values the config
        names on their faces; the state comes from :state: -- a restart
        reader -- or is uniform at the config's initial conditions, and is
        made consistent."""
        self.config = config
        self.mixture = Mixture(config["mcPhysics"])
        super().__init__(self.mixture.speciesNames)

        # what a step does
        self.graphs = {
            "consistify": Graph.consistify(config, self.mixture),
            "consistifyFromPrims": Graph.consistify(
                config, self.mixture, fromPrims=True
            ),
            "rhs": Graph.rhs(config),
        }
        # what every kernel runs with: the species data, the block table each
        # block fills in as it allocates, and the jit; the last two learn the
        # halo depth once every kernel is known
        self.thtrdat = thtrdat(self.mixture)
        self.table = Table()
        self.jit = Jit(self.mixture.ns)

        # what calls kernels: the halo exchange, the graphs, the integrator,
        # and the checks the run makes
        self.communicator = Communicator(self.table, self.thtrdat)
        for graph in self.graphs.values():
            graph.bind(self.table, self.thtrdat, self.communicator)
        self.integrator = getIntegrator(config["timeIntegration"]["integrator"])(
            self.table, self.thtrdat, self.graphs, config
        )
        self.controller = getController(config["timeIntegration"])
        checks = [
            BoundKernel(self.table, self.thtrdat, f"utils/{name}.cpp")
            for name in ("allFinite", "CFLmax")
        ]
        owners = [self.communicator, *self.graphs.values(), self.integrator]
        kernels = [k for o in owners for k in o.kernels] + checks
        # every kernel the case calls, by the name it calls it
        self.kernels = {k.role: k for k in kernels}

        # the halo is as deep as the widest stencil among them
        self.ng = self.table.ng = self.jit.ng = max(k.stencil for k in kernels)
        self.jit.compile(kernels)

        # the blocks, and what follows from them; the hooks find their faces
        # before the first step, and again whenever a face changes
        self.facesChanged = True
        mesh.fill(self)
        # the grid file this case came from, if it came from one
        self.meshFile = getattr(mesh, "fileName", None)
        self.setBlockCommunication()
        self.unifyGrid()
        self.computeMetrics()
        self._applyBcValues()

        # the state, and what follows from it
        if state is None:
            self._setUniformState()
        else:
            state.fill(self)
            # a result holds the interior; the halos start as the nearest cell
            for blk in self.blocks:
                blk.fillHaloWithNearest("q")
        self.consistifyFromPrims()
        # what the integrator keeps beyond the state
        if state is None:
            self.integrator.initialize()
        else:
            self.integrator.restore(state.found)

        # the step being taken
        self.dt = config["timeIntegration"]["dt"]
        self.plugins = pluginsOf(self, config)

    ###########################################################################
    # The kernels
    ###########################################################################
    def __getattr__(self, name):
        # a kernel is reached by the name it was registered under
        kernels = self.__dict__.get("kernels", {})
        if name in kernels:
            return kernels[name]
        raise AttributeError(name)

    ###########################################################################
    # The boundary conditions, on the faces
    ###########################################################################
    def _applyBcValues(self):
        """Make every named face what its config entry says, and give it the
        values that entry sets."""
        bcValues = self.config["bcValues"]
        for blk, face in self.faces():
            if face.bcName is None:
                continue
            if face.bcName not in bcValues:
                raise pgConfigError(
                    "bcValues",
                    face.bcName,
                    f"block {blk.nblki} face {face.nface} carries this name,"
                    f" which the config says nothing about; it knows {sorted(bcValues)}.",
                )
            entry = bcValues[face.bcName]
            if "bcType" not in entry:
                raise pgConfigError("bcValues", face.bcName, "names no bcType.")
            face.bcType = entry["bcType"]
            if face.bc.values:
                face.bc.setValues(entry)

    def applyBcs(self, hook, faces=None):
        """One hook on the given faces, or on every face that has it, the way
        the step does: after euler the faces' state follows, and a condition
        with more to say once it has a density says it."""
        self._connectBcs()
        graph = self.graphs["consistify" if hook in ("euler", "postEos") else "rhs"]
        node = graph.hook(hook)
        if faces is None:
            table = node.table
        else:
            faces = [f for f in faces if hook in f.bc.hooks]
            table = Table.ofFaces(faces, hook) if faces else None
        if table is None:
            return
        node.run(self.tme, table)
        # a hook that sets primitives leaves the faces' state to follow
        if hook in ("euler", "postEos"):
            self.stateFromPrims(table=table)
        if hook == "euler":
            self.applyBcs("postEos", faces)

    def _setUniformState(self):
        """Every cell of q at the config's initial conditions."""
        ic = self.config["initialConditions"]
        Y = ic["Y"]
        unknown = sorted(set(Y) - set(self.speciesNames))
        if unknown:
            raise pgConfigError(
                "initialConditions",
                "Y",
                f"names {unknown}, which are not species of this mixture: "
                f"{self.speciesNames}.",
            )
        if sum(Y.values()) > 1.0 + 1e-12:
            raise pgConfigError(
                "initialConditions", "Y", f"sums to {sum(Y.values())}, more than one."
            )
        values = [ic[k] for k in ("p", "u", "v", "w", "T")]
        values += [Y.get(name, 0.0) for name in self.speciesNames[:-1]]
        for blk in self.blocks:
            q = blk.q.get()
            q[...] = values
            blk.q.set(q)

    ###########################################################################
    # Stepping
    ###########################################################################
    def _connectBcs(self):
        """The faces each hook runs over, whenever a face's condition or
        arrays have changed since the hooks last looked."""
        if self.facesChanged:
            faces = [f for _, f in self.faces()]
            for graph in self.graphs.values():
                graph.connect(faces)
            self.facesChanged = False

    def consistify(self):
        self._connectBcs()
        self.graphs["consistify"].run(self.tme)

    def consistifyFromPrims(self):
        self._connectBcs()
        self.graphs["consistifyFromPrims"].run(self.tme)

    def step(self, dt):
        """One step of the integrator, and the clock and the count move on."""
        self._connectBcs()
        self.dt = dt
        report = "report" in self.plugins and self.plugins["report"].due(self)
        self.integrator.step(self.tme, dt, report)
        self.nrt += 1
        self.tme += dt

    def run(self):
        """The case, start to finish: every step the config asks for, each
        sized by the controller, and every plugin acting as often as it
        says."""
        try:
            for _ in range(self.config["simulation"]["niter"]):
                self.step(self.controller.dt(self))
                for plugin in self.plugins.values():
                    if plugin.due(self):
                        plugin(self)
        finally:
            for plugin in self.plugins.values():
                plugin.finalize(self)

    ###########################################################################
    # What the ranks agree on
    ###########################################################################
    @property
    def myCells(self):
        """This rank's cell count, as the one-entry array the reductions take."""
        return np.array([sum(b.nCells for b in self.blocks)], dtype=np.int32)

    @property
    def numCells(self):
        comm, rank, size = getCommRankSize()
        n = self.myCells
        comm.Allreduce(MPI.IN_PLACE, n, op=MPI.SUM)
        return n[0]

    @property
    def loadEfficiency(self):
        """How far the slowest rank's cell count is from the mean, in percent,
        and which rank it is; None on the other ranks."""
        comm, rank, size = getCommRankSize()
        mine = self.myCells
        recv = np.empty(size, dtype=np.int32) if rank == 0 else None
        comm.Gather(mine, recv, root=0)
        if rank != 0:
            return None, None
        return np.mean(recv) / np.max(recv) * 100.0, np.argmax(recv)

    def maxCFL(self):
        """The max acoustic, convective and combined CFL speeds over every
        rank; the convective floor keeps a step sized from it finite in a
        quiescent field."""
        comm, rank, size = getCommRankSize()
        cfl = np.zeros(3)
        self.CFLmax(cfl=cfl)
        cfl[1] = max(cfl[1], 1e-16)
        comm.Allreduce(MPI.IN_PLACE, cfl, op=MPI.MAX)
        return cfl

    ###########################################################################
    # The grid, once every block is on its rank
    ###########################################################################
    def generateHalo(self):
        for blk in self.blocks:
            blk.generateHalo()

    def unifyGrid(self):
        self.generateHalo()

        # Lets just be clean and create the edges and corners
        for _ in range(3):
            self.communicator.exchange("nodes")

        for blk in self.blocks:
            periodic = [f for f in blk.faces if f.periodicRotation is not None]
            if not periodic:
                continue
            nodes = blk.nodes.get()
            for face in periodic:
                R, t = face.periodicRotation, face.periodicTranslation
                # the halo came from the partner, so it lands where the
                # transform puts it, turned or moved or both
                h = face.halo(nodes)
                h[...] = (h.reshape(-1, 3) @ R.T + t).reshape(h.shape)
            blk.nodes.set(nodes)

    def setBlockCommunication(self):
        for blk in self.blocks:
            blk.setBlockCommunication()
        self.communicator.connect(self.faces())

    def __repr__(self):
        string = f"  Blocks: {len(self.blocks)} of {self.totalBlocks}\n"
        string += f"  Species: {self.mixture.speciesNames}\n"
        string += f"  Time Integrator: {self.integrator.integratorName}\n"
        string += f"  Step Size: {self.controller.name}\n"
        string += f"  Equation of State: {self.config['mcPhysics']['eos']}\n"
        if not self.config["RHS"]["diffusion"]:
            string += "  Diffusion terms not solved for\n"
        for graph in self.graphs.values():
            string += "\n".join("  " + line for line in repr(graph).split("\n")) + "\n"
        return string
