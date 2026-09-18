"""How each step is sized. A controller is the integrator's: it gives the
size of the next step, and declares only the kernel it alone needs."""

import numpy as np
from mpi4py import MPI

from ..kernel import CellCenterKernel
from ..misc import getCommRankSize


class BaseController:
    """What sizes an integrator's steps: stepSize(), and what it alone
    needs declared."""

    # what the config calls it
    name = None

    def __init__(self, solver):
        self.solver = solver
        self.config = solver.config

    def arrays(self):
        """Gives the block arrays this controller needs; none, for most."""
        return {}

    def declKernels(self):
        """Gives the kernels this controller calls, by tag; none, for
        most."""
        return {}

    def stepSize(self):
        """Gives the size of the next step."""
        raise NotImplementedError


class Fixed(BaseController):
    """Every step the config's dt."""

    name = "fixed"

    def stepSize(self):
        return self.config["timeIntegration"]["dt"]


class CFL(BaseController):
    """Each step as large as the config's max CFL allows, up to its max dt."""

    name = "cfl"

    def declKernels(self):
        return {"CFLmax": CellCenterKernel("utils/CFLmax.cpp")}

    def stepSize(self):
        """Gives the config's CFL over the max combined speed on any rank,
        which the speed of sound keeps finite."""
        ti = self.config["timeIntegration"]
        cfl = np.zeros(3, self.solver.backend.fpdtype)
        self.solver.launch("CFLmax", "interior", cfl=cfl)
        getCommRankSize()[0].Allreduce(MPI.IN_PLACE, cfl, op=MPI.MAX)
        return min(ti["maxCFL"] / cfl[2], ti["maxDt"])
