"""Compute over a grid: the backend, the kernels compiled for a physics,
the tables, the halo exchanges, the state and the integrator, made in the
order solver.__init__ lays out by interrogating the simulator and the
integrator for what they need.

What is not the solver's: the physics (the simulator's), taking and
sizing a step (the integrator's), and printing (the report plugin's)."""

import numpy as np
from functools import cached_property

from mpi4py import MPI

from .restart import restart
from .solverBlock import solverBlock
from ..backend import Backend
from ..files.configFile import pgConfigError
from ..graph import BCNode, ExchangeGraphs, Graph, LaunchNode
from ..kernel import CellCenterKernel, HaloExchangeKernel
from .. import integrators
from .. import multiBlock
from ..misc import getCommRankSize
from ..plugins import getPlugins
from ..simulator import getSimulator


class solver(restart):
    """A runnable case: the blocks with the arrays a physics and its
    integrator need, every kernel compiled, the tables and tilings the
    launches run over, the halo exchanges, the boundary conditions applied
    on the faces, the graphs the simulator and the integrator say, and
    the integrator that runs them. The one handle a plugin, a writer, a
    script or a test holds."""

    ###########################################################################
    # Making a case
    ###########################################################################
    def __init__(self, config, mesh, state=None):
        """Makes a case, ready to step: the simulator and the integrator
        the config names; every kernel they call, compiled; the blocks from
        :mesh: -- a mesher or a grid reader, anything that fills a
        multiBlock -- connected, haloed and metricked, with the boundary
        values the config names on their faces, and the tables, exchanges
        and graphs over them; the state from :state: -- a restart reader --
        or uniform at the config's initial conditions, made consistent."""
        self.config = config
        # the mesh this case came from: a mesher, or a grid file's reader
        # with the partition it took, which the results name
        self.mesh = mesh
        self.simulator = getSimulator(config)
        super().__init__(self.simulator.primVars)
        self.exportVars = self.simulator.exportVars
        # its arrays are made where the kernels run
        self.backend = Backend.fromRuntime(config)
        self.integrator = integrators.getIntegrator(config, self)
        # the plugins say what they need with the rest, and start once built
        self.plugins = getPlugins(config)
        self._declArrays()
        self._declKernels()
        self._jit()
        mesh.fill(self)
        self._declComm()
        self._unifyGrid()
        self.computeMetrics()
        # after the metrics, which a boundary's values may be made from, and
        # before the graphs, which launch by the boundaries the faces carry
        self._setBcs()
        self._buildGraphs()
        self._setState(state)
        for plugin in self.plugins.values():
            plugin.start(self)

    def _newBlock(self, nblki):
        return solverBlock(nblki, self)

    ###########################################################################
    # What the simulator and the integrator ask for
    ###########################################################################
    def _declArrays(self):
        """Declares on every block the metrics and the arrays the
        simulation and the integrator need."""
        for name in self.simulator.metrics:
            self.declMetric(name)
        arrays = {**self.simulator.arrays(), **self.integrator.arrays()}
        for plugin in self.plugins.values():
            arrays.update(plugin.arrays())
        for name, spec in arrays.items():
            self.declArray(name, **spec)

    def _declKernels(self):
        """Takes the simulator's and the integrator's kernels by tag, with
        the solver's own: the copy of a block array, and the pack, unpack
        and direct fill of a halo exchange."""
        self.kernels = {
            **self.simulator.declKernels(),
            **self.integrator.declKernels(),
            "copy": CellCenterKernel("utils/copy.cpp"),
            "pack": HaloExchangeKernel("utils/extractSendBuffer.cpp"),
            "unpack": HaloExchangeKernel("utils/placeRecvBuffer.cpp"),
            # the partner's face and how it turns its plane, read off the
            # face it meets
            "directHaloFill": HaloExchangeKernel(
                "utils/directHaloFill.cpp",
                columns={
                    "partnerNface": "nface@neighborFace",
                    "partnerTranspose": "transpose@neighborFace",
                    "partnerFlip0": "flip0@neighborFace",
                    "partnerFlip1": "flip1@neighborFace",
                },
            ),
        }
        for plugin in self.plugins.values():
            for tag, kernel in plugin.declKernels().items():
                if tag in self.kernels:
                    raise ValueError(f"{plugin.name}: the tag {tag} is taken")
                self.kernels[tag] = kernel

    def _jit(self):
        """Compiles every kernel with what the simulator bakes in, and the
        halo as deep as the widest stencil among them."""
        self.ng = max(k.stencil for k in self.everyKernel)
        self.jit = self.backend.jit(self.ng, *self.simulator.bakes())
        self.jit.compile(self.everyKernel)

    @property
    def everyKernel(self):
        """Lists every kernel under the tags, a group's members each."""
        return [k for v in self.kernels.values() for stage in v.stages for k in stage]

    ###########################################################################
    # The grid, once every block is on its rank
    ###########################################################################
    def _setBcs(self):
        """Makes every named block face what the config's bcValues entry of
        its name says it is, and gives it the values the entry sets; a face
        the grid did not name stays what the grid made it."""
        bcValues = self.config["bcValues"]
        for blk, face in self.faces():
            name = face.bcName
            if name is None:
                continue
            if name not in bcValues:
                raise pgConfigError(
                    "bcValues",
                    name,
                    f"block {blk.nblki} face {face.nface} carries this name, which"
                    f" the config says nothing about; it knows {sorted(bcValues)}.",
                )
            entry = bcValues[name]
            if "bcType" not in entry:
                raise pgConfigError("bcValues", name, "names no bcType.")
            face.bcType = entry["bcType"]
            bc = self.simulator.bcBase.named(face.bcType)(face)
            if bc.values:
                bc.setValues(entry)

    def _unifyGrid(self):
        """Fills every block's node halo from its neighbors, and moves a
        periodic halo to where its transform puts it."""
        self.generateHalo()
        for g in ExchangeGraphs("unify", "nodes").bind(*self.means):
            g.run()
        self.movePeriodicHalos()

    ###########################################################################
    # Communication: a halo exchange per exchanged array
    ###########################################################################
    def _declComm(self):
        """Makes a halo exchange, of the kind the config names, for every
        array declared exchanged: the whole halo, ng, unless the
        declaration says fewer planes."""
        k, HaloExchange = self.kernels, multiBlock.getHaloExchange(self.config)
        self.exchanges = {
            name: HaloExchange(
                name,
                self.blockFaceArrayTable,
                self.connOnRankFaces,
                self.connOffRankFaces,
                self.ng if depth is True else depth,
                k["pack"],
                k["unpack"],
                k["directHaloFill"],
            )
            for name, depth in self.exchangedArrays.items()
        }

    ###########################################################################
    # Tables and launches
    ###########################################################################
    @cached_property
    def blockFaceArrayTable(self):
        """Gives the table of every block face on this rank, the backend's,
        made once over the faces as they come."""
        return self.backend.arrayTable([face for _, face in self.faces()])

    @property
    def blockArrayTable(self):
        """Gives the table of every block on this rank, the backend's."""
        return self.backend.arrayTable(self.blocks)

    def launch(self, tag, rangeName, **given):
        """Launches the kernels under :tag: now, outside any graph, over the
        named range of every block: a reduction read on the host, or a
        one-off."""
        return LaunchNode(self.kernels[tag], rangeName).bind(*self.means).run(**given)

    ###########################################################################
    # The graphs, made from what the simulator and the integrator say
    ###########################################################################
    @property
    def means(self):
        """Gives what a node is bound to, in the order bind takes: the
        block and block face tables, and the exchanges by array."""
        return (self.blockArrayTable, self.blockFaceArrayTable, self.exchanges)

    def _buildGraphs(self):
        """Makes every graph the simulator and the integrator say, by
        stage, bound to this rank's means, with what the plugins add at the
        end of a stage after it."""
        specs = {
            **self.simulator.graphs(self.integrator.dtOnDevice),
            **self.integrator.graphs(),
        }
        for plugin in self.plugins.values():
            for stage, nodes in plugin.after().items():
                if stage not in specs:
                    raise ValueError(f"{plugin.name}: no stage {stage} to follow")
                specs[stage] = [*specs[stage], Graph(f"after {stage}", nodes)]
        means = self.means
        self.graphs = {
            stage: [g for spec in stageSpecs for g in spec.bind(*means)]
            for stage, stageSpecs in specs.items()
        }

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
        self._forget("prims")

    ###########################################################################
    # The state
    ###########################################################################
    def _setState(self, state):
        """Sets the state from the primitive vector a fresh case or a result
        gives, and what the integrator keeps beyond it."""
        if state is None:
            values = self.simulator.initialState()
            for blk in self.blocks:
                prims = blk.prims.get()
                prims[...] = values
                blk.prims.set(prims)
        else:
            state.fill(self)
            # a result holds the interior; the halos start as the nearest cell
            for blk in self.blocks:
                blk.fillHaloWithNearest("prims")
        self.consistifyFromPrims()
        if state is None:
            self.integrator.initialize()
        else:
            self.integrator.restore(state.found)

    def exportData(self, blk, names):
        """Gives the named variables of a block as host arrays over every
        cell, halos included, as the simulator derives them from the
        state."""
        return self.simulator.exportData(blk, names)

    def setPrimitives(self, primitives):
        """Sets the state from :primitives:, a host array per block of the
        primitive vector over every cell, halos included."""
        for blk, values in zip(self.blocks, primitives):
            blk.replace("prims", values)
        self._forget("prims")
        self.consistifyFromPrims()

    ###########################################################################
    # Named block arrays, on every block at once
    ###########################################################################
    def copyArray(self, dst, src):
        """Copies one block array into another, every element."""
        self.launch("copy", "full", A=dst, B=src)

    def swapArrays(self, a, b):
        """Trades two named block arrays' places on every block, so nothing
        is copied to shift a state back one step; the tables read both
        again. A captured graph keeps the arrays it captured, whatever
        their names now, so whoever swaps under one runs a set captured
        under each assignment."""
        for blk in self.blocks:
            x, y = getattr(blk, a), getattr(blk, b)
            setattr(blk, a, y), setattr(blk, b, x)
        self._forget(a)
        self._forget(b)

    def _forget(self, name):
        """Drops what the tables hold of one block array: it is another
        array now, or none, and no captured graph launches over it."""
        self.blockArrayTable.forget(name)
        self.blockFaceArrayTable.forget(name)
        self.blockFaceArrayTable.forget(f"{name}@neighborFace")

    ###########################################################################
    # The boundary conditions, on the faces
    ###########################################################################
    def applyBcs(self, bcHook, where="all"):
        """Runs one bcHook now on the block faces :where: names -- all,
        onRank or offRank -- the way the step does; a bcHook no kernel of
        the case has runs nothing."""
        BCNode(self.kernels[f"bcs {bcHook}"], where).bind(*self.means).run()

    ###########################################################################
    # Running, counts and the banner
    ###########################################################################
    def run(self):
        """Runs the case start to finish, the integrator's loop."""
        self.integrator.run()

    @property
    def _myCells(self):
        """This rank's cell count, as the one-entry array the reductions take."""
        return np.array([sum(b.nCells for b in self.blocks)], dtype=np.int32)

    @property
    def numCells(self):
        comm, rank, size = getCommRankSize()
        n = self._myCells
        comm.Allreduce(MPI.IN_PLACE, n, op=MPI.SUM)
        return n[0]

    @property
    def loadEfficiency(self):
        """How far the slowest rank's cell count is from the mean, in percent,
        and which rank it is; None on the other ranks."""
        comm, rank, size = getCommRankSize()
        mine = self._myCells
        recv = np.empty(size, dtype=np.int32) if rank == 0 else None
        comm.Gather(mine, recv, root=0)
        if rank != 0:
            return None, None
        return np.mean(recv) / np.max(recv) * 100.0, np.argmax(recv)

    def __repr__(self):
        string = f"  Blocks: {len(self.blocks)} of {self.totalBlocks}\n"
        if self.mesh.partitionName:
            string += (
                f"  Partition: {self.mesh.partitionName} (ranks x ranks per node)\n"
            )
        string += self.simulator.report() + self.integrator.report()
        for stage in self.graphs.values():
            for graph in stage:
                string += (
                    "\n".join("  " + line for line in repr(graph).split("\n")) + "\n"
                )
        return string
