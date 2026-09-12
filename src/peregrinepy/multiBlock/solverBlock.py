import numpy as np

from ..abi import DeviceArray
from .haloMixin import HaloMixin
from .restartBlock import restartBlock
from .solverMetricsMixin import SolverMetricsMixin
from .solverFace import solverFace
from ..integrators import getIntegrator


class solverBlock(restartBlock, SolverMetricsMixin, HaloMixin):
    def __init__(self, nblki, spNames, ng, config):
        restartBlock.__init__(self, nblki, spNames, ng)
        self.ne = 5 + self.ns - 1
        self.config = config

        #######################################################################
        # Grid metrics only a solver needs
        #######################################################################
        self.declare("J", kind="cell")
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

        integrator = getIntegrator(config["timeIntegration"]["integrator"])
        self.declare(
            *(f"Q{n}" for n in range(integrator.nStorage)),
            kind="cell",
            components=self.ne,
        )
        if integrator.stepType == "dualTime":
            self.declare("Qn", "Qnm1", kind="cell", components=self.ne)
            self.declare("dtau", kind="cell")

    def setExtents(self, ni, nj, nk):
        """A face is shaped by the block it bounds, so it learns how big that
        block is at the same moment the block does."""
        for face in self.faces:
            face.setExtents(ni, nj, nk, self.ne)
        super().setExtents(ni, nj, nk)

    def allocate(self):
        """A solver block's arrays live where the kernels run. The host sees
        one through get() and writes one through set()."""
        for name in self.declared:
            shape = self.shapeOf(name)
            current = getattr(self, name)
            if current is not None and current.shape == shape:
                continue
            setattr(self, name, DeviceArray(shape))

    def hostCopy(self, name):
        return getattr(self, name).get()

    def store(self, name, values):
        getattr(self, name).set(values)

    def fillHaloWithNearest(self, name):
        a = getattr(self, name).get()
        ng = self.ng
        a[0:ng] = a[[ng]]
        a[-ng::] = a[[-ng - 1]]
        a[:, 0:ng] = a[:, [ng]]
        a[:, -ng::] = a[:, [-ng - 1]]
        a[:, :, 0:ng] = a[:, :, [ng]]
        a[:, :, -ng::] = a[:, :, [-ng - 1]]
        getattr(self, name).set(a)

    def _newFace(self, nface):
        return solverFace(nface, self.ng)

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
