import numpy as np
from mpi4py import MPI

from .restart import restart
from .solverBlock import solverBlock
from ..graph import Graph
from ..integrators import getIntegrator
from ..jit import Jit
from ..kernel import BoundKernel
from ..mixture import Mixture
from ..mpiComm import Communicator
from ..mpiComm.mpiUtils import getCommRankSize
from ..table import Table
from ..files.configFile import pgConfigError
from ..writers.writeDualTimeQnm1 import writeDualTimeQnm1
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
        self.mixture = Mixture(config["mcPhysics"], root=config["io"]["inputDir"])
        super().__init__(self.mixture.speciesNames)

        # what a step does
        self.graphs = {
            "consistify": Graph.consistify(config, self.mixture),
            "consistifyFromPrims": Graph.consistify(
                config, self.mixture, fromPrims=True
            ),
            "rhs": Graph.rhs(config),
        }
        # time integrator time
        self.titme = 0.0

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
        self.integrator = getIntegrator(config["timeIntegration"]["integrator"])(self)
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
        self.setBlockCommunication()
        self.unifyGrid()
        self.computeMetrics()
        self._applyBcValues()

        # the state, and what follows from it
        if state is None:
            self._setUniformState()
        else:
            state.fill(self)
        self.consistifyFromPrims()

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
        node.run(self.titme, table)
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
        self.graphs["consistify"].run(self.titme)

    def consistifyFromPrims(self):
        self._connectBcs()
        self.graphs["consistifyFromPrims"].run(self.titme)

    def RHS(self):
        self._connectBcs()
        self.graphs["rhs"].run(self.titme)

    def step(self, dt):
        self.integrator.step(dt)

    def advance(self, dt):
        """The step is taken: the clock and the count move on."""
        self.nrt += 1
        self.tme += dt
        self.titme = self.tme

    def run(self):
        """The case, start to finish: every step the config asks for, with
        the results written, the state checked and the coprocessor called as
        often as it says."""
        from ..coproc import coprocessor
        from ..writers import RestartWriter

        comm, rank, size = getCommRankSize()
        config = self.config
        io = config["io"]
        writer = RestartWriter(
            self,
            quiet=True,
            path=io["resultsDir"],
            gridPath=f"../{io['gridDir']}",
            precision="single",
        )
        coproc = coprocessor(self)
        self.integrator.initialize()

        niterOut, niterPrint = io["niterOut"], io["niterPrint"]
        checkNan = config["simulation"]["checkNan"]
        for _ in range(config["simulation"]["niter"]):
            dt, CFLmaxA, CFLmaxC, CFLmax = self.dtMaxCFL()
            if self.nrt % niterPrint == 0 and rank == 0:
                print(
                    f" >>> --------- nrt: {self.nrt:<6} ---------- <<<\n",
                    f"    tme: {self.tme:.6E} s\n"
                    f"     dt : {dt:.6E} s\n"
                    f"     MAX CFL       : {CFLmax*dt:.3f}\n"
                    f"         Acoustic  : {CFLmaxA*dt:.3f}\n"
                    f"         Convective: {CFLmaxC*dt:.3f}\n"
                    " >>> -------------------------------- <<<\n",
                )

            self.step(dt)

            if self.nrt % niterOut == 0:
                if rank == 0:
                    print("Saving results.\n")
                writer.write(self)
                if self.integrator.stepType == "dualTime":
                    writeDualTimeQnm1(self, path=io["resultsDir"])

            if checkNan and self.nrt % checkNan == 0 and self.checkForNan() > 0:
                self.nrt = 99999999
                writer.write(self)
                comm.Barrier()
                if rank == 0:
                    print("Nan/inf detected. Aborting.")
                break

            coproc(self)

        coproc.finalize()

    ###########################################################################
    # What the ranks agree on
    ###########################################################################
    @property
    def numCells(self):
        comm, rank, size = getCommRankSize()
        n = np.array(
            [sum((b.ni - 1) * (b.nj - 1) * (b.nk - 1) for b in self.blocks)],
            dtype=np.int32,
        )
        comm.Allreduce(MPI.IN_PLACE, n, op=MPI.SUM)
        return n[0]

    @property
    def loadEfficiency(self):
        """How far the slowest rank's cell count is from the mean, in percent,
        and which rank it is; None on the other ranks."""
        comm, rank, size = getCommRankSize()
        mine = np.array(
            [sum((b.ni - 1) * (b.nj - 1) * (b.nk - 1) for b in self.blocks)],
            dtype=np.int32,
        )
        recv = np.empty(size, dtype=np.int32) if rank == 0 else None
        comm.Gather(mine, recv, root=0)
        if rank != 0:
            return None, None
        return np.mean(recv) / np.max(recv) * 100.0, np.argmax(recv)

    def dtMaxCFL(self):
        """The time step, and the max acoustic, convective and combined CFL
        speeds over every rank; the convective floor keeps the step finite in
        a quiescent field."""
        comm, rank, size = getCommRankSize()
        cfl = np.zeros(3)
        self.CFLmax(cfl=cfl)
        cfl[1] = max(cfl[1], 1e-16)
        comm.Allreduce(MPI.IN_PLACE, cfl, op=MPI.MAX)
        ti = self.config["timeIntegration"]
        dt = (
            min(ti["maxCFL"] / cfl[2], ti["maxDt"])
            if ti["variableTimeStep"]
            else ti["dt"]
        )
        return dt, cfl[0], cfl[1], cfl[2]

    def checkForNan(self):
        """How many ranks hold a non-finite conserved value; each such rank
        logs where its are."""
        comm, rank, size = getCommRankSize()
        abort = np.array([0], np.int32)
        abort[0] = not self.allFinite()
        if abort[0]:
            for blk in self.blocks:
                Q = blk.Q.get()
                ng = blk.ng
                nans = np.where(
                    np.sum(np.isnan(Q[ng:-ng, ng:-ng, ng:-ng, :]), axis=-1) > 0
                )
                if len(nans[0]) == 0:
                    continue
                cells = blk.cells.get()[ng:-ng, ng:-ng, ng:-ng]
                with open(f"nans_{blk.nblki}.log", "w") as f:
                    f.write(f"Nan Detection Log: Block {blk.nblki}\\n")
                    for x, y, z in cells[nans]:
                        f.write(f"x = {x} y = {y} z = {z}\\n")
        comm.Allreduce(MPI.IN_PLACE, abort, op=MPI.SUM)
        return abort[0]

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
                for s0 in face.s0_:
                    # the halo came from the partner, so it lands where the
                    # transform puts it, turned or moved or both
                    p = nodes[s0]
                    nodes[s0] = (p.reshape(-1, 3) @ R.T + t).reshape(p.shape)
            blk.nodes.set(nodes)

    def setBlockCommunication(self):
        for blk in self.blocks:
            blk.setBlockCommunication()
        self.communicator.connect(self.faces())

    def __repr__(self):
        string = f"  Blocks: {len(self.blocks)} of {self.totalBlocks}\n"
        string += f"  Species: {self.mixture.speciesNames}\n"
        string += f"  Time Integrator: {self.integrator.integratorName}\n"
        string += f"  Equation of State: {self.config['mcPhysics']['eos']}\n"
        if not self.config["RHS"]["diffusion"]:
            string += "  Diffusion terms not solved for\n"
        for graph in self.graphs.values():
            string += "\n".join("  " + line for line in repr(graph).split("\n")) + "\n"
        return string
