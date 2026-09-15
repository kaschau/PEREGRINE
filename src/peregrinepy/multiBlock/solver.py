"""The system of a case: its blocks and their arrays, the species data, the
tables, every kernel by tag, the flows, the boundary conditions and the
exchange, made in the order solver.__init__ lays out.

What is not the solver's: taking a step and sizing it (the stepper's and the
controller's, composed onto it by integrators.getSolver), stage storage (a
stepper declares its own), printing (the report plugin's), and any kernel
that only a plugin or a controller calls (they declare their own)."""

import numpy as np
from mpi4py import MPI

from .restart import restart
from .solverBlock import solverBlock
from ..backend import runtimeBackend
from ..bcs import bcTypesWith
from ..graph import Graph
from ..jit import Jit
from ..kernel import (
    BlockFaceKernel,
    CellCenterKernel,
    CellFaceKernel,
    HaloExchangeKernel,
    KernelGroup,
)
from ..mixture import Mixture
from ..mpiComm import HaloExchange
from ..mpiComm.mpiUtils import getCommRankSize
from ..plugins import pluginsOf
from ..table import Table
from ..files.configFile import pgConfigError


class solver(restart):
    """A runnable case: the blocks, the mixture and its species data, the
    step's graphs, the boundary conditions, and every kernel the config asks
    for, compiled and bound. integrators.getSolver composes a controller
    and a stepper onto this, which is how a case is made: the stepper takes
    a step and says what it keeps, the controller sizes each step and runs
    the case."""

    def _newBlock(self, nblki):
        return solverBlock(nblki, self)

    def __init__(self, config, mesh, state=None):
        """A case, ready to step. The step's graphs and every kernel they call
        are compiled for the config; the blocks come from :mesh: -- a mesher
        or a grid reader, anything that fills a multiBlock -- and are
        connected, haloed and metricked, with the boundary values the config
        names on their faces; the state comes from :state: -- a restart
        reader -- or is uniform at the config's initial conditions, and is
        made consistent."""
        self.config = config
        self._refuse(config)
        self.mixture = Mixture(config["mcPhysics"])
        super().__init__(self.mixture.speciesNames)
        # its arrays are made where the kernels run
        self.backend = runtimeBackend()
        self.ne = 5 + self.mixture.ns - 1

        # every array a block holds beyond the restart's, and every kernel the
        # case calls by tag: the base's, the stepper's and the controller's;
        # nothing about order, which the graphs hold
        self.declareArrays()
        self.kernels = {}
        self.declareKernels()
        kernels = self._everyKernel()
        # the halo is as deep as the widest stencil among them; the table and
        # the jit follow, and each kernel is bound to what it runs over
        self.ng = max(k.stencil for k in kernels)
        tileSize = config["RHS"]["tileSize"]
        self.table = Table(self.blocks, tileSize, self.backend)
        self.jit = Jit(self.mixture.ns, self.ng, tileSize, self.mixture.tables())
        self.jit.compile(kernels)
        for k in kernels:
            if isinstance(k, (CellCenterKernel, CellFaceKernel)):
                k.bind(self.table)

        # what calls them: the halo exchange and the step's flows
        self.haloExchange = HaloExchange(
            self.kernels["pack"], self.kernels["unpack"], tileSize, self.backend
        )
        self.graphs = {
            "consistify": Graph.consistify(self),
            "consistifyFromPrims": Graph.consistify(self, fromPrims=True),
            "rhs": Graph.rhs(self),
        }
        for graph in self.graphs.values():
            graph.check(self.arrays)

        # the blocks, and what follows from them; the bcs find their faces
        # before the first step, and again whenever a face changes
        self.facesChanged = True
        self._blockFaceTables = {}
        mesh.fill(self)
        # the grid file this case came from, if it came from one, and the
        # partition of it this run took
        self.meshFile = getattr(mesh, "fileName", None)
        self.partition = getattr(mesh, "partitionName", None)
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
        # what the stepper keeps beyond the state
        if state is None:
            self.initialize()
        else:
            self.restore(state.found)

        # the step being taken
        self.dt = config["timeIntegration"]["dt"]
        self.plugins = pluginsOf(self, config)

    ###########################################################################
    # The kernels
    ###########################################################################
    @staticmethod
    def _refuse(config):
        """What the flows do not describe yet is refused, not run."""
        rhs, mc, ti = config["RHS"], config["mcPhysics"], config["timeIntegration"]
        for key in ("shockHandling", "secondaryAdvFlux", "switchAdvFlux", "subgrid"):
            if rhs[key] is not None:
                raise pgConfigError(key, rhs[key], "not until the composition round")
        if config["viscousSponge"]["spongeON"]:
            raise pgConfigError(
                "viscousSponge", True, "not until the composition round"
            )
        if mc["chemistry"]:
            raise pgConfigError("chemistry", True, "not until the composition round")
        if rhs["primaryAdvFlux"] is None:
            raise pgConfigError("primaryAdvFlux", None, "a case has a primary flux")
        if mc["eos"] not in ("cpg", "tpg", "realGas"):
            raise pgConfigError("eos", mc["eos"])
        if ti["integrator"] == "dualTime":
            if mc["eos"] not in ("cpg", "tpg"):
                raise pgConfigError(
                    "dualTime", mc["eos"], "only cpg and tpg are supported"
                )
            if ti["controller"] != "fixed":
                raise pgConfigError(
                    "dualTime", ti["controller"], "only a fixed time step is supported"
                )
            if ti["pseudoIntegrator"] == "dualTime":
                raise pgConfigError(
                    "pseudoIntegrator",
                    "dualTime",
                    "the pseudo time scheme is Runge-Kutta",
                )

    def declareArrays(self):
        """Every array a solver's block holds beyond its grid and state: the
        metrics, the conserved state and its derivative, the face fluxes,
        the thermodynamic state, and with diffusion the gradients and the
        transport properties. A stepper adds what it keeps through super()."""
        ne, ns = self.ne, self.mixture.ns
        self.declareArray("Jinv", kind="cell")
        self.declareArray("dIJK", kind="cell", components=3)
        # cell center transformation metrics
        self.declareArray("dENCdxyz", kind="cell", components=(3, 3))
        for axis in "ijk":
            self.declareArray(f"{axis}Faces", kind=f"{axis}face", components=3)
            self.declareArray(f"{axis}S", kind=f"{axis}face", components=3)
            self.declareArray(f"{axis}F", kind=f"{axis}face", components=ne)
        self.declareArray("Q", kind="cell", components=ne)
        self.declareArray("dQ", kind="cell", components=ne)
        self.declareArray("qh", kind="cell", components=5 + ns)
        if self.config["RHS"]["diffusion"]:
            self.declareArray("grads", kind="cell", components=(ne, 3))
            self.declareArray("qt", kind="cell", components=2 + ns)

    def declareKernels(self):
        """The kernels every case calls, each under its tag: the linear
        combinations of block arrays, the equation of state, the fluxes and
        the apply, the transport, the exchange, and every bcType's body at
        every bcHook the case has. A stepper and a controller add their own
        through super()."""
        rhs, mc = self.config["RHS"], self.config["mcPhysics"]
        k = self.kernels
        k["axpby"] = CellCenterKernel("utils/axpby.cpp")
        k["axpbypcz"] = CellCenterKernel("utils/axpbypcz.cpp")
        k["stateFromCons"] = CellCenterKernel(f"thermo/{mc['eos']}FromCons.cpp")
        k["stateFromPrims"] = CellCenterKernel(f"thermo/{mc['eos']}FromPrims.cpp")
        k["primaryAdvFlux"] = KernelGroup(
            [
                CellFaceKernel(f"advFlux/{rhs['primaryAdvFlux']}.cpp", d)
                for d in range(3)
            ]
        )
        bcHooks = ["euler"]
        if rhs["diffusion"]:
            k["trans"] = CellCenterKernel(
                f"transport/{self.mixture.transportKernel}.cpp"
            )
            k["dqdxyz"] = CellCenterKernel("utils/dq2FD.cpp")
            k["diffFlux"] = KernelGroup(
                [CellFaceKernel("diffFlux/alphaDampingFlux.cpp", d) for d in range(3)]
            )
            bcHooks += ["preDqDxyz", "postDqDxyz"]
        k["applyFlux"] = CellCenterKernel("utils/applyFlux.cpp")
        k["pack"] = HaloExchangeKernel("utils/extractSendBuffer.cpp")
        k["unpack"] = HaloExchangeKernel("utils/placeRecvBuffer.cpp")
        # every bcType's body at each bcHook, kept on its own, so the
        # kernels are fixed and only the faces are found at a run
        for bcHook in bcHooks:
            for bcType in bcTypesWith(bcHook):
                k[f"{bcType}@{bcHook}"] = BlockFaceKernel(bcType, bcHook)

    def _everyKernel(self):
        """Every kernel under the tags, a group's members each."""
        return [
            k
            for v in self.kernels.values()
            for k in (v.kernels if isinstance(v, KernelGroup) else (v,))
        ]

    def __getattr__(self, name):
        # a kernel is reached by its tag
        kernels = self.__dict__.get("kernels", {})
        if name in kernels:
            return kernels[name]
        # a property that raised an AttributeError of its own lands here too;
        # run it again so that error is the one seen
        attr = getattr(type(self), name, None)
        if isinstance(attr, property):
            return attr.fget(self)
        raise AttributeError(name)

    ###########################################################################
    # Named block arrays, on every block at once
    ###########################################################################
    def copyArray(self, dst, src):
        """One block array's interior into another's: axpby with a leading
        zero, which writes without reading."""
        self.axpby(A=dst, a=0.0, b=1.0, B=src)

    def swapArrays(self, a, b):
        """The two named block arrays trade places on every block, so nothing
        is copied to shift a state back one step; the table reads both
        again."""
        for blk in self.blocks:
            x, y = getattr(blk, a), getattr(blk, b)
            setattr(blk, a, y), setattr(blk, b, x)
        self.table.forget(a)
        self.table.forget(b)

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

    def blockFaceTables(self, bcHook, faces=None):
        """The block faces with a bcType at :bcHook:, grouped by bcType into
        a table each: every face's, kept until a face changes, or :faces:'."""
        if faces is not None:
            return self._tablesOf(bcHook, faces)
        if self.facesChanged:
            self._blockFaceTables = {}
            self.facesChanged = False
        if bcHook not in self._blockFaceTables:
            self._blockFaceTables[bcHook] = self._tablesOf(
                bcHook, [f for _, f in self.faces()]
            )
        return self._blockFaceTables[bcHook]

    def _tablesOf(self, bcHook, faces):
        groups = {}
        for f in faces:
            if bcHook in f.bc.bcHooks():
                groups.setdefault(f.bc.bcType, []).append(f)
        return {
            t: Table(fs, self.table.tileSize, self.backend) for t, fs in groups.items()
        }

    def applyBcs(self, bcHook, faces=None):
        """One bcHook on the given faces, or on every face that has it, the
        way the step does: after euler the faces' state follows. A bcHook
        the case's flow does not have runs nothing."""
        graph = self.graphs["consistify" if bcHook == "euler" else "rhs"]
        node = graph.bcs(bcHook)
        if node is None:
            return
        tables = self.blockFaceTables(bcHook, faces)
        node.run(tables)
        if bcHook == "euler":
            graph.faceState.run(tables)

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
    # Stepping: the three verbs a stepper and the tests use, each one flow
    ###########################################################################
    def rhs(self):
        self.graphs["rhs"].run()

    def consistify(self):
        self.graphs["consistify"].run()

    def consistifyFromPrims(self):
        self.graphs["consistifyFromPrims"].run()

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
            self.haloExchange.exchange("nodes")

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
        self.haloExchange.connect(self.faces())

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

    def __repr__(self):
        string = f"  Blocks: {len(self.blocks)} of {self.totalBlocks}\n"
        if self.partition:
            string += f"  Partition: {self.partition} (ranks x ranks per node)\n"
        string += f"  Species: {self.mixture.speciesNames}\n"
        ti = self.config["timeIntegration"]
        string += f"  Time Integrator: {self.stepperName}\n"
        if ti["integrator"] == "dualTime":
            string += f"  Pseudo Time: {ti['pseudoIntegrator']}, {ti['subIterations']} steps\n"
        string += f"  Step Size: {self.controllerName}\n"
        string += f"  Equation of State: {self.config['mcPhysics']['eos']}\n"
        if not self.config["RHS"]["diffusion"]:
            string += "  Diffusion terms not solved for\n"
        for graph in self.graphs.values():
            string += "\n".join("  " + line for line in repr(graph).split("\n")) + "\n"
        return string
