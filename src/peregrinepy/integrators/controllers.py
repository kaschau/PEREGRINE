"""How the run proceeds in time. A controller is a mixin composed onto the
solver by getSolver: it sizes each step, takes it through the stepper with
the clock and the count moving on, and runs the case step by step."""

import numpy as np
from mpi4py import MPI

from ..kernel import CellCenterKernel
from ..mpiComm.mpiUtils import getCommRankSize


class BaseController:
    """What a controller is: stepSize(), step(dt) with the clock and the
    count, and run() with the plugins. What it is not: it takes no stage
    (the stepper's advance), declares only the kernel it alone needs, and
    prints nothing."""

    # what the config calls it
    controllerName = None

    def stepSize(self):
        """The size of the next step."""
        raise NotImplementedError

    def step(self, dt):
        """One step of :dt:: the stepper's stages, and the clock and the
        count move on."""
        self.dt = dt
        self.advance(dt)
        self.nrt += 1
        self.tme += dt

    def run(self):
        """The case, start to finish: every step the config asks for, each
        sized here, and every plugin acting as often as it says."""
        try:
            for _ in range(self.config["simulation"]["niter"]):
                dt = self.stepSize()
                for plugin in self.plugins.values():
                    plugin.before(self, dt)
                self.step(dt)
                for plugin in self.plugins.values():
                    if plugin.due(self):
                        plugin(self)
        finally:
            for plugin in self.plugins.values():
                plugin.finalize(self)


class Fixed(BaseController):
    """Every step the config's dt."""

    controllerName = "fixed"

    def stepSize(self):
        return self.config["timeIntegration"]["dt"]


class CFL(BaseController):
    """Each step as large as the config's max CFL allows, up to its max dt."""

    controllerName = "cfl"

    def declareKernels(self):
        super().declareKernels()
        self.kernels["CFLmax"] = CellCenterKernel("utils/CFLmax.cpp")

    def stepSize(self):
        """The config's CFL over the max combined speed on any rank, which
        the speed of sound keeps finite."""
        ti = self.config["timeIntegration"]
        cfl = np.zeros(3)
        self.launch("CFLmax", "interior", cfl=cfl)
        getCommRankSize()[0].Allreduce(MPI.IN_PLACE, cfl, op=MPI.MAX)
        return min(ti["maxCFL"] / cfl[2], ti["maxDt"])
