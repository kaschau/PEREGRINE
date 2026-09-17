"""The system of a case: its blocks and their arrays, the species data, the
two tables, every kernel by tag, the halo exchanges, the boundary
conditions, and the step's graphs, made in the order solver.__init__ lays
out.

What is not the solver's: taking a step and sizing it (the stepper's and the
controller's, composed onto it by integrators.getSolver), stage storage (a
stepper declares its own), printing (the report plugin's), and any kernel
that only a plugin or a controller calls (they declare their own)."""

import numpy as np
from mpi4py import MPI

from .arrays import CellCenterArray, CellFaceArray
from .restart import restart
from .solverBlock import solverBlock
from ..backend import Backend
from ..bcs import bcTypesWith, getBc
from ..graph import CollectiveLaunchNode, Graph
from ..kernel import (
    CellCenterKernel,
    CellFaceKernel,
    HaloExchangeKernel,
    UnorderedKernelGroup,
)
from ..mixture import Mixture
from ..mpiComm import BaseHaloExchange
from ..mpiComm.mpiUtils import getCommRankSize
from ..plugins import pluginsOf
from ..files.configFile import pgConfigError


class solver(restart):
    """A runnable case: the blocks, the mixture and its species data, every
    kernel the config asks for, the tables and tilings the launches run
    over, the halo exchanges, the boundary conditions, and the step's
    graphs. integrators.getSolver composes a controller and a stepper onto
    this, which is how a case is made: the stepper takes a step and says
    what it keeps, the controller sizes each step and runs the case."""

    def _newBlock(self, nblki):
        return solverBlock(nblki, self)

    def __init__(self, config, mesh, state=None):
        """Makes a case, ready to step. Every kernel is compiled for the
        config; the blocks come from :mesh: -- a mesher or a grid reader,
        anything that fills a multiBlock -- and are connected, haloed and
        metricked, with the boundary values the config names on their
        faces; the state comes from :state: -- a restart reader -- or is
        uniform at the config's initial conditions, and is made
        consistent."""
        self.config = config
        self._refuse(config)
        self.mixture = Mixture(config["mcPhysics"])
        super().__init__(self.mixture.speciesNames)
        # its arrays are made where the kernels run
        self.backend = Backend.fromRuntime(config)
        self.ne = 5 + self.mixture.ns - 1

        # every array a block holds beyond the restart's, and every kernel the
        # case calls by tag: the base's, the stepper's and the controller's
        self.declareArrays()
        self.kernels = {}
        self.declareKernels()
        kernels = self._everyKernel()
        # the halo is as deep as the widest stencil among them; the jit follows
        self.ng = max(k.stencil for k in kernels)
        self.jit = self.backend.jit(self.ng, self.mixture, config["mcPhysics"])
        self.jit.compile(kernels)

        # the two tables every launch runs over, over the blocks and their
        # block faces as the mesh fills them in; then the blocks, wired to
        # their neighbors, and the halo exchanges, fixed from here
        self.blockFaces = []
        self.blockTable = self.backend.table(self.blocks)
        self.blockFaceTable = self.backend.table(self.blockFaces)
        mesh.fill(self)
        self.blockFaces.extend(face for _, face in self.faces())
        # the grid file this case came from, if it came from one, and the
        # partition of it this run took
        self.meshFile = getattr(mesh, "fileName", None)
        self.partition = getattr(mesh, "partitionName", None)
        self.setBlockCommunication()
        self.tradingFaces, self.localFaces, self.remoteFaces = self._sortFaces()
        self.hereFaces = [f for f in self.blockFaces if f not in self.remoteFaces]
        exchangeKind = BaseHaloExchange.fromConfig(config)
        self.exchanges = {
            name: exchangeKind(
                name,
                self.blockFaceTable,
                self.tradingFaces,
                self.localFaces,
                self.remoteFaces,
                depth,
            )
            for name, depth in self.exchanged()
        }
        self.unifyGrid()
        self.computeMetrics()
        self._applyBcValues()
        # the step's graphs, from what is now fixed
        self.graphs = {}
        self.declareGraphs()

        # the state, from the primitive vector a fresh case or a result
        # gives, and what follows from it
        if state is None:
            self._setUniformState()
        else:
            state.fill(self)
            # a result holds the interior; the halos start as the nearest cell
            for blk in self.blocks:
                blk.fillHaloWithNearest("prims")
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
    # What the case declares: arrays, kernels, exchanges, graphs
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
        if rhs["primaryAdvFlux"] in ("fourthOrderKEEP", "muscl2hllc", "muscl2rusanov"):
            raise pgConfigError(
                "primaryAdvFlux",
                rhs["primaryAdvFlux"],
                "the hand-unrolled schemes return as a muscl, limiter and riemann"
                " solver composition",
            )
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
        """Declares every array a solver's block holds beyond its grid and
        the primitive vector it starts from: the metrics, the conserved
        state and its derivative, the cell-face fluxes, the thermodynamic
        state -- p and T in q, what the eos keeps in qh -- and with
        diffusion the gradients and the transport properties. A stepper
        adds what it keeps through super()."""
        ne, ns = self.ne, self.mixture.ns
        self.declareArray("Jinv", CellCenterArray)
        self.declareArray("dIJK", CellCenterArray, components=3)
        # cell center transformation metrics
        self.declareArray("dENCdxyz", CellCenterArray, components=(3, 3))
        for axis, d in enumerate("ijk"):
            self.declareArray(f"{d}Faces", CellFaceArray, components=3, axis=axis)
            self.declareArray(f"{d}S", CellFaceArray, components=3, axis=axis)
            self.declareArray(f"{d}F", CellFaceArray, components=ne, axis=axis)
        self.declareArray("Q", CellCenterArray, components=ne)
        self.declareArray("dQ", CellCenterArray, components=ne)
        self.declareArray("q", CellCenterArray, components=2)
        self.declareArray(
            "qh", CellCenterArray, components=self.mixture.eos.qhComponents(ns)
        )
        if self.config["RHS"]["diffusion"]:
            # the velocity, T and Y(0 .. ns - 2): nothing diffuses on pressure
            self.declareArray("grads", CellCenterArray, components=(ne - 1, 3))
            self.declareArray("qt", CellCenterArray, components=2 + ns)

    def declareKernels(self):
        """Declares the kernels every case calls, each under its tag: the
        copy and linear combinations of block arrays, the equation of state,
        the fluxes and the apply, the transport, the pack and unpack of the
        exchanges, and every bcType's body at every bcHook the case has. A
        stepper and a controller add their own through super()."""
        rhs, mc = self.config["RHS"], self.config["mcPhysics"]
        k = self.kernels
        k["copy"] = CellCenterKernel("utils/copy.cpp")
        k["axpby"] = CellCenterKernel("utils/axpby.cpp")
        k["axpbypcz"] = CellCenterKernel("utils/axpbypcz.cpp")
        # the equation of state is compiled into these by the jit
        k["stateFromCons"] = CellCenterKernel("thermo/stateFromCons.cpp")
        k["stateFromPrims"] = CellCenterKernel("thermo/stateFromPrims.cpp")
        # a scheme's directions write their own flux: no order between them
        k["primaryAdvFlux"] = UnorderedKernelGroup(
            [
                CellFaceKernel(f"advFlux/{rhs['primaryAdvFlux']}.cpp", d)
                for d in range(3)
            ]
        )
        bcHooks = ["euler"]
        if rhs["diffusion"]:
            # the species diffusion model is compiled into it by the jit
            k["trans"] = CellCenterKernel(f"transport/{mc['trans']}.cpp")
            k["dqdxyz"] = CellCenterKernel("utils/dq2FD.cpp")
            k["diffFlux"] = UnorderedKernelGroup(
                [CellFaceKernel("diffFlux/alphaDampingFlux.cpp", d) for d in range(3)]
            )
            bcHooks += ["preDqDxyz", "postDqDxyz"]
        k["applyFlux"] = CellCenterKernel("utils/applyFlux.cpp")
        k["pack"] = HaloExchangeKernel("utils/extractSendBuffer.cpp")
        k["unpack"] = HaloExchangeKernel("utils/placeRecvBuffer.cpp")
        # every bcType's body at each bcHook, its own kernel: bc.cpp compiled
        # with the bcType's header forced in
        for bcHook in bcHooks:
            for bcType in bcTypesWith(bcHook):
                k[f"{bcType}@{bcHook}"] = CellCenterKernel(
                    "boundaryConditions/bc.cpp",
                    defines=(f"PG_BCTYPE={bcType}", f"PG_BCHOOK={bcHook}"),
                    includes=(getBc(bcType).header(),),
                )

    def exchanged(self):
        """Names the arrays whose halos are exchanged, with the planes each
        trades: the nodes and the state ng deep, the gradients one."""
        pairs = [("nodes", self.ng), ("Q", self.ng)]
        if self.config["RHS"]["diffusion"]:
            pairs.append(("grads", 1))
        return pairs

    def declareGraphs(self):
        """Declares the step's graphs, each stage a list of device graphs
        cut where a message is waited on: making the state consistent is
        the Q exchange around the equation of state, the boundary
        conditions' euler bcHook and the transport, the halos a message
        brought done again after it lands; the right-hand side is the
        gradient exchange around the advective and diffusive fluxes, then
        the apply. A stepper adds its own through super()."""
        viscous = self.config["RHS"]["diffusion"]
        Q = self.exchanges["Q"]
        stage = [
            Graph("consistify: pack Q", before=[Q.expect], after=[Q.copyOut]),
            Graph("consistify: state", after=[Q.send, Q.receive]),
            Graph("consistify: remote halos", after=[Q.sent]),
        ]
        stage[0].add(self.packNode("Q"))
        stage[0].add(self.unpackNode("Q", "local"))
        stage[1].add(self.launchNode("stateFromCons", "all"))
        stage[1].add(self.bcNode("euler", self.hereFaces, "here"))
        if viscous:
            stage[1].add(self.launchNode("trans", "all"))
        stage[2].add(self.unpackNode("Q", "remote"))
        stage[2].add(self.bcNode("euler", self.remoteFaces, "remote"))
        stage[2].add(self.redoNode("stateFromCons"))
        if viscous:
            stage[2].add(self.redoNode("trans"))
        self.graphs["consistify"] = stage

        if not viscous:
            rhs = Graph("rhs")
            rhs.add(self.launchNode("primaryAdvFlux", "interior"))
            rhs.add(self.launchNode("applyFlux", "interior"))
            self.graphs["rhs"] = [rhs]
            return
        grads = self.exchanges["grads"]
        stage = [
            Graph("rhs: gradients", before=[grads.expect], after=[grads.copyOut]),
            Graph("rhs: fluxes", after=[grads.send, grads.receive]),
            Graph("rhs: remote faces", after=[grads.sent]),
        ]
        stage[0].add(self.bcNode("preDqDxyz", self.blockFaces))
        stage[0].add(self.launchNode("dqdxyz", "interior"))
        stage[0].add(self.packNode("grads"))
        stage[0].add(self.unpackNode("grads", "local"))
        stage[1].add(self.launchNode("primaryAdvFlux", "interior"))
        stage[1].add(self.bcNode("postDqDxyz", self.hereFaces, "here"))
        stage[1].add(self.launchNode("diffFlux", "interior"))
        stage[2].add(self.unpackNode("grads", "remote"))
        stage[2].add(self.bcNode("postDqDxyz", self.remoteFaces, "remote"))
        stage[2].add(self.redoNode("primaryAdvFlux", "diffFlux"))
        stage[2].add(self.launchNode("applyFlux", "interior"))
        self.graphs["rhs"] = stage

    def _everyKernel(self):
        """Lists every kernel under the tags, a group's members each."""
        return [k for v in self.kernels.values() for stage in v.stages for k in stage]

    ###########################################################################
    # Launches: a kernel over a table and a tiling
    ###########################################################################
    def tiling(self, kernel, over):
        """Gives the tiling of the block table a kernel's kind of item takes
        over the named canonical range -- full, all, interior -- made once
        and kept."""
        components = self._componentsOf(kernel)
        axis = getattr(kernel, "direction", None)
        key = (kernel.items, axis, components, over)
        if key not in self.blockTable.tilings:
            ranges = [
                (n, start, extent, components)
                for n, blk in enumerate(self.blocks)
                for start, extent in getattr(kernel.rangeOf(blk), over)()
            ]
            self.blockTable.tile(key, ranges, kernel.tileKind)
        return self.blockTable.tilings[key]

    def _componentsOf(self, kernel):
        """Resolves the components an item of this kernel is of: a count,
        or ne less a count."""
        components = kernel.components
        if isinstance(components, str):
            _, _, less = components.partition("-")
            return self.ne - (int(less) if less else 0)
        return components

    def launch(self, tag, over, **given):
        """Runs the kernel under :tag: now over the named range of the block
        table; :given: supplies its scalars and any array by name."""
        kernel = self.kernels[tag]
        return kernel(self.blockTable, self.tiling(kernel, over), **given)

    def launchNode(self, tag, over, **fixed):
        """Makes a node launching the kernels under :tag:, each over the
        named range of the block table."""
        stages = [
            [(k, self.blockTable, self.tiling(k, over)) for k in stage]
            for stage in self.kernels[tag].stages
        ]
        return CollectiveLaunchNode(tag, stages, **fixed)

    def faceTiling(self, key, faces, boxesOf, items="cells"):
        """Gives the tiling of the block-face table under :key:, made once
        from :boxesOf:(face) over these faces and kept."""
        if key not in self.blockFaceTable.tilings:
            index = {id(f): n for n, f in enumerate(self.blockFaces)}
            ranges = [
                (index[id(f)], start, extent, 1)
                for f in faces
                for start, extent in boxesOf(f)
            ]
            self.blockFaceTable.tile(key, ranges, items)
        return self.blockFaceTable.tilings[key]

    def bcNode(self, bcHook, faces, where=""):
        """Makes the node of the boundary conditions at one bcHook over
        these block faces: every bcType with a kernel there over the halo
        cells behind the faces carrying it, as one stage since their faces
        are disjoint. A hook that follows an unpack is done in two: the
        faces whose halos are here, then the ones a message brings (a
        periodic across ranks), once it has landed."""
        stage = []
        for tag, kernel in self.kernels.items():
            if not tag.endswith(f"@{bcHook}"):
                continue
            bcType = tag.removesuffix(f"@{bcHook}")
            mine = [f for f in faces if f.bcType == bcType]
            tiling = self.faceTiling(("halo", bcType, where), mine, kernel.behind)
            stage.append((kernel, self.blockFaceTable, tiling))
        return CollectiveLaunchNode(f"bcs {bcHook} {where}".strip(), [stage])

    def redoNode(self, *tags):
        """Makes the node launching the kernels under :tags: again over what
        a message brought: a cell kernel over the halo cells behind the
        remote block faces, a flux scheme over the plane of cell faces on
        them, in the order given."""
        remote = self.remoteFaces
        stages = []
        for tag in tags:
            for stage in self.kernels[tag].stages:
                stages.append(
                    [
                        (k, self.blockFaceTable, self._remoteTiling(k, remote))
                        for k in stage
                    ]
                )
        return CollectiveLaunchNode(f"{' '.join(tags)} remote", stages)

    def _remoteTiling(self, kernel, remote):
        """Gives the tiling of what this kernel does again over the remote
        block faces that concern it."""
        key = ("remote", kernel.items, getattr(kernel, "direction", None))
        faces = [f for f in remote if kernel.concerns(f)]
        return self.faceTiling(key, faces, kernel.behind)

    def packNode(self, name):
        """Makes the node packing one array's halos for its neighbors, over
        every trading block face."""
        exchange = self.exchanges[name]
        stage = [(self.kernels["pack"], self.blockFaceTable, exchange.tilings["pack"])]
        return CollectiveLaunchNode(f"pack {name}", [stage], **exchange.packArgs)

    def unpackNode(self, name, which):
        """Makes the node unpacking one array's halos behind the block faces
        met on this rank (local) or on another (remote)."""
        exchange = self.exchanges[name]
        stage = [(self.kernels["unpack"], self.blockFaceTable, exchange.tilings[which])]
        return CollectiveLaunchNode(
            f"unpack {name} {which}", [stage], **exchange.unpackArgs
        )

    def exchange(self, name):
        """Fills every block's halos of one array now, the graph's steps in
        a row: what the grid's nodes need at setup."""
        ex = self.exchanges[name]
        ex.expect()
        self.packNode(name).run()
        self.unpackNode(name, "local").run()
        ex.copyOut()
        ex.send()
        ex.receive()
        self.unpackNode(name, "remote").run()
        ex.sent()

    def _sortFaces(self):
        """Sorts the block faces once: the ones that trade, the pairs met on
        this rank, and the ones met on another."""
        rank = getCommRankSize()[1]
        trading, local, remote = [], [], []
        for face in self.blockFaces:
            if face.neighbor is None:
                continue
            trading.append(face)
            if face.commRank == rank:
                theirs = self.getBlock(face.neighbor).getFace(face.neighborNface)
                local.append((face, theirs))
            else:
                remote.append(face)
        return trading, local, remote

    ###########################################################################
    # Named block arrays, on every block at once
    ###########################################################################
    def copyArray(self, dst, src):
        """Copies one block array into another, every element."""
        self.launch("copy", "full", A=dst, B=src)

    def swapArrays(self, a, b):
        """Trades two named block arrays' places on every block, so nothing
        is copied to shift a state back one step; the tables read both
        again."""
        for blk in self.blocks:
            x, y = getattr(blk, a), getattr(blk, b)
            setattr(blk, a, y), setattr(blk, b, x)
        self.forget(a)
        self.forget(b)

    def forget(self, name):
        """Drops what the tables hold of one block array: it is another
        array now, and no graph launches over these."""
        self.blockTable.forget(name)
        self.blockFaceTable.forget(name)

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

    def applyBcs(self, bcHook, faces=None):
        """Runs one bcHook now on the given block faces, or on every one,
        the way the step does; a bcHook no graph of the case has runs
        nothing."""
        faces = self.blockFaces if faces is None else faces
        self.bcNode(bcHook, faces, tuple(id(f) for f in faces)).run()

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
            prims = blk.prims.get()
            prims[...] = values
            blk.prims.set(prims)

    ###########################################################################
    # Stepping: the three verbs a stepper and the tests use
    ###########################################################################
    def rhs(self):
        for graph in self.graphs["rhs"]:
            graph.run()

    def consistify(self):
        for graph in self.graphs["consistify"]:
            graph.run()

    def consistifyFromPrims(self):
        """Makes the state from every block's primitive vector, then
        everything derived from it. The vector is held only for this: a
        case starts from it, a test sets it through setPrimitives, and it
        is released once the state is made."""
        self.launch("stateFromPrims", "all")
        self.consistify()
        for blk in self.blocks:
            blk.prims = None
        self.forget("prims")

    def setPrimitives(self, primitives):
        """Sets the state from :primitives:, a host array per block of p, u,
        v, w, T, Y(0 .. ns - 2) over every cell, halos included."""
        for blk, values in zip(self.blocks, primitives):
            blk.replace("prims", values)
        self.consistifyFromPrims()

    ###########################################################################
    # The grid, once every block is on its rank
    ###########################################################################
    def generateHalo(self):
        for blk in self.blocks:
            blk.generateHalo()

    def unifyGrid(self):
        """Fills every block's node halo from its neighbors, and moves a
        periodic halo to where its transform puts it."""
        self.generateHalo()
        self.exchange("nodes")
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
        for stage in self.graphs.values():
            for graph in stage:
                string += (
                    "\n".join("  " + line for line in repr(graph).split("\n")) + "\n"
                )
        return string
