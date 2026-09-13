import numpy as np
from mpi4py import MPI

from ..mpiComm.mpiUtils import getCommRankSize
from .explicit import BaseIntegrator


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


class dualTime(BaseIntegrator):
    integratorName = "dualTime"
    stepType = "dualTime"
    # the inner pseudo time loop is rk3 like
    nStorage = 2
    # the kernels the pseudo time stages call
    sources = tuple(
        f"timeIntegration/{k}.cpp"
        for k in ("dQdt", "localDtau", "dTrk3s1", "dTrk3s2", "dTrk3s3", "invertDQ")
    ) + ("utils/axpby.cpp", "utils/residual.cpp")
    # pseudo time steps per physical step
    subIterations = 20

    def __init__(self, table, thtrdat, graphs, config):
        super().__init__(table, thtrdat, graphs, config)
        self.viscous = config["RHS"]["diffusion"]
        self.ne = 5 + thtrdat.ns - 1

    def _stage(self, stage, tme, dt):
        """One pseudo time rk3 stage: the RHS at its time, the physical time
        derivative, the preconditioned update."""
        self.rhs.run(tme)
        self.dQdt(dt=dt)
        self.invertDQ(dt=dt, viscous=self.viscous)
        stage()
        self.consistifyFromPrims.run(tme)

    def _copy(self, dst, src):
        views = self.table.views
        self.axpby(A=views(dst), a=0.0, b=1.0, B=views(src))

    def step(self, tme, dt, report=False):
        """Qn and Qnm1 are the two states before :tme:; the pseudo time loop
        converges the state at :tme: + :dt:, then the three shift."""
        comm, rank, size = getCommRankSize()
        for n in range(self.subIterations):
            self.localDtau(viscous=self.viscous)
            self._stage(self.dTrk3s1, tme, dt)
            self._stage(self.dTrk3s2, tme + dt, dt)
            self._stage(self.dTrk3s3, tme + dt / 2.0, dt)

            if report:
                resid = np.zeros((2, self.ne))
                self.residual(rMax=resid[0], rSum=resid[1])
                comm.Allreduce(MPI.IN_PLACE, resid[0, :], op=MPI.MAX)
                comm.Allreduce(MPI.IN_PLACE, resid[1, :], op=MPI.SUM)
                resid[1, :] = np.sqrt(resid[1, :])
                if rank == 0:
                    printResidual(resid[1, :], n, self.ne)

        self._copy("Qnm1", "Qn")
        self._copy("Qn", "Q")

    def initialize(self):
        """A fresh case begins at rest in time: both earlier states are this one."""
        self._copy("Qn", "Q")
        self._copy("Qnm1", "Q")

    def restore(self, blocks, nrt, path):
        """Qnm1 as the result was written with, or this state when the result
        carries none; Qn is this state."""
        for blk in blocks:
            ng = blk.ng
            try:
                with open(f"{path}/Qnm1.{nrt:08d}.{blk.nblki:06d}.npy", "rb") as f:
                    Qnm1 = blk.Qnm1.get()
                    Qnm1[ng:-ng, ng:-ng, ng:-ng, :] = np.load(f)
                    blk.Qnm1.set(Qnm1)
            except FileNotFoundError:
                blk.Qnm1.copyFrom(blk.Q)
        self._copy("Qn", "Q")

    def writeState(self, blocks, nrt, path):
        """Qnm1 beside the result, so a restart from it steps as this run did."""
        for blk in blocks:
            ng = blk.ng
            with open(f"{path}/Qnm1.{nrt:08d}.{blk.nblki:06d}.npy", "wb") as f:
                np.save(f, blk.Qnm1.get()[ng:-ng, ng:-ng, ng:-ng, :])
