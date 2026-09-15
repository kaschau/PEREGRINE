import numpy as np
from mpi4py import MPI

from ..mpiComm.mpiUtils import getCommRankSize
from .rungeKutta import RKIntegrator, rk3


def printResidual(resid, nrt, ne):
    if nrt == 0:
        string = " SubIter      p          u          v          w          T"
        if ne > 5:
            string += "        Y(1) ... Y(NS-1)"
        print(string)
    string = f"{nrt+1:8d}"
    for n in range(ne):
        string += f" {resid[n]: 1.3E}"
    print(string)


class dualTime(RKIntegrator):
    """Each physical step converged in pseudo time: rk3's stages on the
    primitives, each stepping by the preconditioned increment invertDQ
    leaves in dQ, with the physical time derivative as a source."""

    integratorName = "dualTime"
    stepType = "dualTime"
    nStorage = 1
    sources = (
        RKIntegrator.sources
        + tuple(f"timeIntegration/{k}.cpp" for k in ("dQdt", "localDtau", "invertDQ"))
        + ("utils/residual.cpp",)
    )
    state = "q"
    stages = rk3.stages
    # pseudo time steps per physical step
    subIterations = 20
    stateArrays = ("Qnm1",)

    def __init__(self, table, thtrdat, graphs, config):
        super().__init__(table, thtrdat, graphs, config)
        self.viscous = config["RHS"]["diffusion"]
        self.ne = 5 + thtrdat.ns - 1

    def _stage(self, n, tme, dt):
        """One pseudo time stage: the RHS at its time, the physical time
        derivative, the preconditioned update."""
        frac, *weights = self.stages[n]
        self.rhs.run(tme + frac * dt)
        self.dQdt(dt=dt)
        self.invertDQ(dt=dt, viscous=self.viscous)
        self.combine(*weights, dt=1.0, first=n == 0)
        self.consistifyFromPrims.run(tme + frac * dt)

    def _copy(self, dst, src):
        self.axpby(A=dst, a=0.0, b=1.0, B=src)

    def step(self, tme, dt, report=False):
        """Qn and Qnm1 are the two states before :tme:; the pseudo time loop
        converges the state at :tme: + :dt:, then the three shift."""
        comm, rank, size = getCommRankSize()
        for n in range(self.subIterations):
            self.localDtau(viscous=self.viscous)
            for n in range(len(self.stages)):
                self._stage(n, tme, dt)

            if report:
                resid = np.zeros((2, self.ne))
                self.residual(rMax=resid[0], rSum=resid[1])
                comm.Allreduce(MPI.IN_PLACE, resid[0, :], op=MPI.MAX)
                comm.Allreduce(MPI.IN_PLACE, resid[1, :], op=MPI.SUM)
                resid[1, :] = np.sqrt(resid[1, :])
                if rank == 0:
                    printResidual(resid[1, :], n, self.ne)

        # the two earlier states shift: Qn's storage becomes Qnm1's, and only
        # the new state is copied
        self.table.swap("Qn", "Qnm1")
        self._copy("Qn", "Q")

    def initialize(self):
        """A fresh case begins at rest in time: both earlier states are this one."""
        self._copy("Qn", "Q")
        self._copy("Qnm1", "Q")

    def restore(self, found):
        """Qn is this state; Qnm1 is what the result carried, or this state
        when it carried none."""
        self._copy("Qn", "Q")
        if "Qnm1" not in found:
            self._copy("Qnm1", "Q")
