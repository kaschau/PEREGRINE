"""How one step is taken. A stepper is a mixin composed onto the solver by
getSolver: it says what it keeps between stages and what a result file
carries of it, and steps the state through the solver's graphs and its own
kernels. Every stepper here is a Runge-Kutta table; dual time wraps one of
them in a pseudo-time loop."""

from functools import cached_property

import numpy as np
from mpi4py import MPI

from ..kernel import CellCenterKernel
from ..misc import subclassWhere
from ..mpiComm.mpiUtils import getCommRankSize


class BaseStepper:
    """What a stepper is: its name in the config, the block arrays it keeps
    between stages and which of them a result carries, the array its stages
    step, and advance(dt). What it is not: it sizes no step and runs no loop
    (the controller's), declares only its own kernels and storage, and
    prints nothing (report() returns text for the report plugin)."""

    # what the config calls it
    stepperName = None
    # the block arrays it keeps between stages, (cell, ne) each
    storage = ()
    # those of them a result file carries
    restartArrays = ()
    # the block array its stages step
    state = "Q"
    # set by the report plugin before a step it will report on, for a
    # stepper that has something to gather during the step
    reportDue = False

    def declareArrays(self):
        super().declareArrays()
        for name in self.storage:
            self.declareArray(name, kind="cell", components=self.ne)

    def initialize(self):
        """What a fresh case does before its first step; nothing, for most."""

    def restore(self, found):
        """What a restarted case does with what the result carried of its
        restart arrays (:found:), and without what it did not; nothing, for
        most."""

    def advance(self, dt):
        """The state at the solver's time to the state :dt: later."""
        raise NotImplementedError

    def report(self):
        """What the stepper has to say about the step just taken, as text for
        the report plugin to print; nothing, for most."""


class rungeKutta(BaseStepper):
    """A step is a fixed list of stages. Each takes the RHS, combines the
    state with the state the step began from and the derivative, and makes
    the state consistent again. The first stage keeps the state it began
    from in Q0."""

    # (fraction of dt, wQ0, wQ, wdQ) per stage: Q = wQ Q + wQ0 Q0 + wdQ dt dQ
    stages = ()

    def advance(self, dt):
        for n, (frac, *weights) in enumerate(self.stages):
            self.rhs()
            self.combine(*weights, dt=dt, first=n == 0)
            self.consistify()

    def combine(self, wQ0, wQ, wdQ, dt, first):
        """One stage's update of the state; a strong stability preserving
        combination unless a scheme says otherwise."""
        Q, dQ = self.state, "dQ"
        if first and "Q0" in self.storage:
            self.copyArray("Q0", Q)
        if wQ0 == 0.0:
            self.axpby(A=Q, a=wQ, b=wdQ * dt, B=dQ)
        else:
            self.axpbypcz(A=Q, a=wQ, b=wQ0, B="Q0", c=wdQ * dt, C=dQ)


class rk1(rungeKutta):
    stepperName = "rk1"
    stages = ((0.0, 0.0, 1.0, 1.0),)


class rk2(rungeKutta):
    stepperName = "rk2"
    storage = ("Q0",)
    stages = ((0.0, 0.0, 1.0, 1.0), (1.0, 0.5, 0.5, 0.5))


class rk3(rungeKutta):
    """
    S. Gottlieb, C.-W Shu. Total variation diminishing Runge-Kutta
    schemes. Mathematics of Computation, 67(221):73-85, 1998.
    """

    stepperName = "rk3"
    storage = ("Q0",)
    stages = (
        (0.0, 0.0, 1.0, 1.0),
        (1.0, 0.75, 0.25, 0.25),
        (0.5, 1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0),
    )


class rk34(rungeKutta):
    stepperName = "rk34"
    storage = ("Q0",)
    stages = (
        (0.0, 0.0, 1.0, 0.5),
        (0.5, 0.0, 1.0, 0.5),
        (1.0, 2.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0),
        (0.5, 0.0, 1.0, 0.5),
    )


class maccormack(rungeKutta):
    """Predictor then corrector, both built at the step's start time. The
    corrector's weights are rk2's second stage; only the time it is evaluated
    at differs."""

    stepperName = "maccormack"
    storage = ("Q0",)
    stages = ((0.0, 0.0, 1.0, 1.0), (0.0, 0.5, 0.5, 0.5))


class rk4(rungeKutta):
    """The classical scheme: every stage steps from the start of the step by
    wdQ dt dQ, and wSum dt dQ of each derivative goes into a running sum,
    which is what the last stage steps by."""

    stepperName = "rk4"
    storage = ("Q0", "Q1")
    # (fraction of dt, wdQ, wSum) per stage; the last has no wdQ of its own
    stages = (
        (0.0, 0.5, 1.0 / 6.0),
        (0.5, 0.5, 1.0 / 3.0),
        (0.5, 1.0, 1.0 / 3.0),
        (1.0, None, 1.0 / 6.0),
    )

    def combine(self, wdQ, wSum, dt, first):
        Q, Q0, S, dQ = self.state, "Q0", "Q1", "dQ"
        if first:
            self.copyArray(Q0, Q)
        self.axpby(A=S, a=0.0 if first else 1.0, b=wSum * dt, B=dQ)
        if wdQ is None:
            self.axpbypcz(A=Q, a=0.0, b=1.0, B=Q0, c=1.0, C=S)
        else:
            self.axpbypcz(A=Q, a=0.0, b=1.0, B=Q0, c=wdQ * dt, C=dQ)


class dualTime(BaseStepper):
    """Each physical step converged in pseudo time: a Runge-Kutta stepper's
    stages on the primitives, each stepping by the preconditioned increment
    invertDQ leaves in dQ, with the physical time derivative as a source.
    Which stepper, and how many pseudo steps, the config says."""

    stepperName = "dualTime"
    restartArrays = ("Qnm1",)
    state = "q"
    # the residual after each pseudo step, gathered when a report is due
    residuals = ()

    @cached_property
    def pseudo(self):
        """The pseudo time scheme the config names: its stages and its
        combination of them."""
        name = self.config["timeIntegration"]["pseudoIntegrator"]
        return subclassWhere(rungeKutta, stepperName=name)

    @property
    def subIterations(self):
        return self.config["timeIntegration"]["subIterations"]

    @property
    def viscous(self):
        return self.config["RHS"]["diffusion"]

    @property
    def storage(self):
        return (*self.pseudo.storage, "Qn", "Qnm1")

    @property
    def stages(self):
        return self.pseudo.stages

    def declareArrays(self):
        super().declareArrays()
        self.declareArray("dtau", kind="cell")
        # the preconditioning reads the transport properties, diffusion or not
        if not self.viscous:
            self.declareArray("qt", kind="cell", components=2 + self.mixture.ns)

    def declareKernels(self):
        super().declareKernels()
        for name in ("dQdt", "localDtau", "invertDQ"):
            self.kernels[name] = CellCenterKernel(f"timeIntegration/{name}.cpp")
        self.kernels["residual"] = CellCenterKernel("utils/residual.cpp")

    def _stage(self, n, dt):
        """One pseudo time stage: the RHS, the physical time derivative, the
        preconditioned update."""
        frac, *weights = self.stages[n]
        self.rhs()
        self.dQdt(dt=dt)
        self.invertDQ(dt=dt, viscous=self.viscous)
        self.pseudo.combine(self, *weights, dt=1.0, first=n == 0)
        self.consistifyFromPrims()

    def advance(self, dt):
        """Qn and Qnm1 are the two states before now; the pseudo time loop
        converges the state :dt: later, then the three shift."""
        comm, rank, size = getCommRankSize()
        self.residuals = []
        for n in range(self.subIterations):
            self.localDtau(viscous=self.viscous)
            for stage in range(len(self.stages)):
                self._stage(stage, dt)
            # the residual is only worth its reduction when it will be reported
            if self.reportDue:
                resid = np.zeros((2, self.ne))
                self.residual(rMax=resid[0], rSum=resid[1])
                comm.Allreduce(MPI.IN_PLACE, resid[1, :], op=MPI.SUM)
                self.residuals.append(np.sqrt(resid[1, :]))

        # the two earlier states shift: Qn's storage becomes Qnm1's, and only
        # the new state is copied
        self.swapArrays("Qn", "Qnm1")
        self.copyArray("Qn", "Q")

    def report(self):
        """The root-sum-square residual after each pseudo time step of the
        step just taken, by equation."""
        header = " SubIter      p          u          v          w          T"
        if self.ne > 5:
            header += "        Y(1) ... Y(NS-1)"
        rows = (
            f"{n + 1:8d}" + "".join(f" {r: 1.3E}" for r in resid)
            for n, resid in enumerate(self.residuals)
        )
        return "\n".join((header, *rows))

    def initialize(self):
        """A fresh case begins at rest in time: both earlier states are this one."""
        self.copyArray("Qn", "Q")
        self.copyArray("Qnm1", "Q")

    def restore(self, found):
        """Qn is this state; Qnm1 is what the result carried, or this state
        when it carried none."""
        self.copyArray("Qn", "Q")
        if "Qnm1" not in found:
            self.copyArray("Qnm1", "Q")
