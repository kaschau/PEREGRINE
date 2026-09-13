import numpy as np

from ..abi import DeviceStorageMixin
from .haloMixin import HaloMixin
from .restartBlock import restartBlock
from .solverMetricsMixin import SolverMetricsMixin
from .solverFace import solverFace


class solverBlock(restartBlock, SolverMetricsMixin, HaloMixin, DeviceStorageMixin):
    def __init__(self, nblki, spNames, ng, config, mb, tableIndex):
        # the solver this block belongs to, and which slot of its table is this block's
        self.mb, self.tableIndex = mb, tableIndex
        restartBlock.__init__(self, nblki, spNames, ng)
        self.ne = 5 + self.ns - 1

        #######################################################################
        # Grid metrics only a solver needs
        #######################################################################
        self.declare("Jinv", kind="cell")
        self.declare("dIJK", kind="cell", components=3)
        # cell center transformation metrics
        self.declare("dENCdxyz", kind="cell", components=(3, 3))
        for axis in "ijk":
            self.declare(f"{axis}Faces", f"{axis}S", kind=f"{axis}face", components=3)

        #######################################################################
        # Solution Variables
        #######################################################################
        # conserved variables, and the primatives restartBlock declared
        self.declare("Q", "dQ", kind="cell", components=self.ne)
        # face fluxes and the switches between them
        for axis in "ijk":
            self.declare(f"{axis}F", kind=f"{axis}face", components=self.ne)
        self.declare("phi", kind="cell", components=3)
        # thermo
        self.declare("qh", kind="cell", components=5 + self.ns)

        # what the case asks for beyond that
        if config["RHS"]["diffusion"]:
            self.declare("grads", kind="cell", components=(self.ne, 3))
            self.declare("qt", kind="cell", components=2 + self.ns)
        if config["mcPhysics"]["chemistry"]:
            self.declare("omega", kind="cell", components=1 + self.ns - 1)

        integrator = mb.integrator
        self.declare(
            *(f"Q{n}" for n in range(integrator.nStorage)),
            kind="cell",
            components=self.ne,
        )
        if integrator.stepType == "dualTime":
            self.declare("Qn", "Qnm1", kind="cell", components=self.ne)
            self.declare("dtau", kind="cell")
            # the preconditioning reads the transport properties when there are any
            if not config["RHS"]["diffusion"]:
                self.declare("qt", kind="cell", components=2 + self.ns)

    def setExtents(self, ni, nj, nk):
        """A face is shaped by the block it bounds, so it learns how big that
        block is at the same moment the block does."""
        for face in self.faces:
            face.setExtents(ni, nj, nk, self.ne)
        super().setExtents(ni, nj, nk)
        self.mb.table.setDims(self.tableIndex, self)

    def _placed(self, name):
        """A new array is a new record in the case's table."""
        self.mb.table.register(self.tableIndex, name, getattr(self, name))

    def _newFace(self, nface):
        return solverFace(nface, self.ng, self)

    @property
    def interior(self):
        """The slice of this block's arrays that is not halo."""
        ng = self.ng
        return np.s_[ng:-ng, ng:-ng, ng:-ng]

    def setBlockCommunication(self):
        for face in self.faces:
            if face.neighbor is None:
                continue
            face.setCommunication(self.nblki)
