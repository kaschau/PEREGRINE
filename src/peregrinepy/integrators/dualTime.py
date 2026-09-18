"""Dual time: each physical step converged in pseudo time by a Runge-Kutta
scheme on the conserved state, each stage stepping by the preconditioned
increment, with the physical time derivative as a source."""

from functools import cached_property

import numpy as np
from mpi4py import MPI

from ..graph import Graph, LaunchNode
from ..kernel import CellCenterKernel
from ..misc import getCommRankSize, subclassWhere
from ..multiBlock.arrays import CellCenterArray
from .base import BaseIntegrator
from .rungeKutta import rungeKutta


class dualTime(BaseIntegrator):
    """Each physical step converged in pseudo time: a Runge-Kutta stepper's
    stages on the conserved state, each stepping by the preconditioned
    increment invertDQ leaves in dQ, with the physical time derivative as a
    source. Which stepper, and how many pseudo steps, the config says."""

    integratorName = "dualTime"
    # a result carries the state one step back; Qn is this one
    restartArrays = ("Qnm1",)
    # the residual after each pseudo step, gathered when a report is due
    residuals = ()
    # Qn and Qnm1 trade names at the end of every step, so the newest
    # state is never copied over the older; a captured graph keeps the
    # arrays it captured, so the pseudo graphs are captured twice, once
    # under each assignment, and this says which set the names match
    bank = 0

    @cached_property
    def pseudo(self):
        """Gives the pseudo time scheme the config names: its stages and its
        combination of them."""
        name = self.config["timeIntegration"]["pseudoIntegrator"]
        return subclassWhere(rungeKutta, integratorName=name)

    @property
    def subIterations(self):
        return self.config["timeIntegration"]["subIterations"]

    @property
    def viscous(self):
        return self.solver.simulation.viscous

    @property
    def storage(self):
        return (*self.pseudo.storage, "Qn", "Qnm1")

    @property
    def stages(self):
        return self.pseudo.stages

    def arrays(self):
        arrays = {**super().arrays(), "dtau": dict(kind=CellCenterArray)}
        # the preconditioning reads the transport properties, diffusion or not
        if not self.viscous:
            ns = self.solver.simulation.mixture.ns
            arrays["qt"] = dict(kind=CellCenterArray, components=2 + ns)
        return arrays

    def declKernels(self):
        k = super().declKernels()
        for name in ("dQdt", "localDtau", "invertDQ"):
            k[name] = CellCenterKernel(f"timeIntegration/{name}.cpp")
        k["residual"] = CellCenterKernel("utils/residual.cpp")
        return k

    def graphs(self):
        """Gives the pseudo time graphs: the local pseudo step, and each
        stage's physical time derivative, preconditioned update and
        combination."""
        dt, viscous, k = self.dtOnDevice, self.viscous, self.kernels
        localDtau = Graph(
            "localDtau", [LaunchNode(k["localDtau"], "interior", viscous=viscous)]
        )
        # the same pseudo graphs twice: one set is captured under each
        # assignment of the names Qn and Qnm1 to their two arrays
        pseudo = {
            f"pseudo {bank}": [
                Graph(
                    f"pseudo {bank} stage {n}",
                    [
                        LaunchNode(k["dQdt"], "interior", dt=dt),
                        LaunchNode(k["invertDQ"], "interior", dt=dt, viscous=viscous),
                    ],
                )
                for n in range(len(self.stages))
            ]
            for bank in (0, 1)
        }
        # the preconditioned increment carries the pseudo step already
        combine = [
            self.pseudo.combineGraph(self, n, *weights, dt=self.one)
            for n, (frac, *weights) in enumerate(self.stages)
        ]
        return {
            **super().graphs(),
            "localDtau": [localDtau],
            **pseudo,
            "combine": combine,
        }

    def advance(self, dt):
        """Converges the state :dt: later in the pseudo time loop, Qn and
        Qnm1 being the two states before now, then shifts the three."""
        solver = self.solver
        comm, rank, size = getCommRankSize()
        self.residuals = []
        graphs = solver.graphs
        pseudos = graphs[f"pseudo {self.bank}"]
        for n in range(self.subIterations):
            for g in graphs["localDtau"]:
                g.run()
            for pseudo, combine in zip(pseudos, graphs["combine"]):
                solver.rhs()
                pseudo.run()
                combine.run()
                solver.consistify()
            # the residual is only worth its reduction when it will be reported
            if self.reportDue:
                resid = np.zeros((2, solver.ne), solver.backend.fpdtype)
                solver.launch("residual", "interior", rMax=resid[0], rSum=resid[1])
                comm.Allreduce(MPI.IN_PLACE, resid[1, :], op=MPI.SUM)
                self.residuals.append(np.sqrt(resid[1, :]))

        # the oldest takes the new state, then the two earlier states trade
        # names: Qn is this one and Qnm1 the one before, whatever the arrays
        solver.copyArray("Qnm1", "Q")
        solver.swapArrays("Qn", "Qnm1")
        self.bank ^= 1

    def stepReport(self):
        """Gives the root-sum-square residual after each pseudo time step of
        the step just taken, by equation."""
        if not self.residuals:
            return None
        header = " SubIter     rho        rhou       rhov       rhow       rhoE"
        if self.solver.ne > 5:
            header += "      rhoY(1) ... rhoY(NS-1)"
        rows = (
            f"{n + 1:8d}" + "".join(f" {r: 1.3E}" for r in resid)
            for n, resid in enumerate(self.residuals)
        )
        return "\n".join((header, *rows))

    def report(self):
        ti = self.config["timeIntegration"]
        return (
            super().report()
            + f"  Pseudo Time: {ti['pseudoIntegrator']}, {ti['subIterations']} steps\n"
        )

    def initialize(self):
        """Starts a fresh case at rest in time: both earlier states are this
        one."""
        self.solver.copyArray("Qn", "Q")
        self.solver.copyArray("Qnm1", "Q")

    def restore(self, found):
        """Takes Qn as this state, and Qnm1 as what the result carried, or
        this state when it carried none."""
        self.solver.copyArray("Qn", "Q")
        if "Qnm1" not in found:
            self.solver.copyArray("Qnm1", "Q")
