import numpy as np

from ..compute import block_
from ..compute.pgkokkos import deep_copy
from .haloMixin import HaloMixin
from .restartBlock import restartBlock
from .solverMetricsMixin import SolverMetricsMixin
from .solverFace import solverFace
from ..integrators import getIntegrator
from ..misc import createViewMirrorArray


class solverBlock(restartBlock, SolverMetricsMixin, HaloMixin):
    blockType = "solver"

    def __init__(self, nblki, spNames, ng, config):
        # must exist before anything forwards to it
        self.cpp = block_()

        restartBlock.__init__(self, nblki, spNames, ng)

        if hasattr(self.cpp, "ns") and self.cpp.ns != self.ns:
            raise ValueError(
                f"ERROR!! You are trying to use {self.ns} species, but pg.compute\n"
                f"    was precompiled for {self.cpp.ns} species."
            )

        self.ne = 5 + self.ns - 1
        self.config = config

        #######################################################################
        # Grid metrics only a solver needs
        #######################################################################
        self.declare("J", "dI", "dJ", "dK", kind="cell")
        # cell center transformation metrics
        self.declare(
            *(f"d{a}d{b}" for a in "ENC" for b in "xyz"),
            kind="cell",
        )
        for axis in "ijk":
            self.declare(
                *(f"{axis}{n}" for n in ("xc", "yc", "zc")),
                *(f"{axis}s{n}" for n in "xyz"),
                f"{axis}S",
                *(f"{axis}n{n}" for n in "xyz"),
                kind=f"{axis}face",
            )

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
            self.declare("dqdx", "dqdy", "dqdz", kind="cell", components=self.ne)
            self.declare("qt", kind="cell", components=2 + self.ns)
        if config["thermochem"]["chemistry"]:
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
        """A solver block's arrays are Kokkos views, with a host mirror the
        numpy array wraps."""
        for name in self.declared:
            createViewMirrorArray(self, name, list(self.shapeOf(name)))

    def fillHaloWithNearest(self, name):
        a = self.array[name]
        ng = self.ng
        a[0:ng] = a[[ng]]
        a[-ng::] = a[[-ng - 1]]
        a[:, 0:ng] = a[:, [ng]]
        a[:, -ng::] = a[:, [-ng - 1]]
        a[:, :, 0:ng] = a[:, :, [ng]]
        a[:, :, -ng::] = a[:, :, [-ng - 1]]

    def _newFace(self, nface):
        return solverFace(nface, self.ng)

    @property
    def interior(self):
        """The slice of this block's arrays that is not halo."""
        ng = self.ng
        return np.s_[ng:-ng, ng:-ng, ng:-ng]

    @property
    def nblki(self):
        return self.cpp.nblki

    @nblki.setter
    def nblki(self, value):
        self.cpp.nblki = value

    @property
    def ni(self):
        return self.cpp.ni

    @ni.setter
    def ni(self, value):
        self.cpp.ni = value

    @property
    def nj(self):
        return self.cpp.nj

    @nj.setter
    def nj(self, value):
        self.cpp.nj = value

    @property
    def nk(self):
        return self.cpp.nk

    @nk.setter
    def nk(self, value):
        self.cpp.nk = value

    @property
    def ng(self):
        return self.cpp.ng

    @ng.setter
    def ng(self, value):
        self.cpp.ng = value

    @property
    def ne(self):
        return self.cpp.ne

    @ne.setter
    def ne(self, value):
        self.cpp.ne = value

    def setBlockCommunication(self):
        for face in self.faces:
            if face.neighbor is None:
                continue
            face.setCommunication(self.nblki)

    def updateDeviceView(self, vars):
        if isinstance(vars, str):
            vars = [vars]
        for var in vars:
            deep_copy(getattr(self.cpp, var), self.mirror[var])

    def updateHostView(self, vars):
        if isinstance(vars, str):
            vars = [vars]
        for var in vars:
            deep_copy(self.mirror[var], getattr(self.cpp, var))
