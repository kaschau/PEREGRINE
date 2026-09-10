from ..compute.timeIntegration import rk4s1, rk4s2, rk4s3, rk4s4
from ..compute.utils import AEQB, axnpby
from ..consistify import consistify
from ..RHS import RHS


def ssp(wQ0, wQ, wdQ, storeQ0=False):
    """A stage of a strong stability preserving scheme,
    Q = wQ Q + wQ0 Q0 + wdQ dt dQ. The stage that begins a step keeps the
    state it began from, since later stages combine with it."""

    def stage(blk, dt):
        if storeQ0:
            AEQB(blk.cpp.Q0, blk.cpp.Q)
        if wQ0 == 0.0:
            axnpby(blk.cpp.Q, wQ, wdQ * dt, blk.cpp.dQ)
        else:
            axnpby(blk.cpp.Q, wQ, wQ0, blk.cpp.Q0, wdQ * dt, blk.cpp.dQ)

    return stage


class BaseExplicit:
    """A step is a fixed list of stages. Each one sets the time the RHS is
    built at, as a fraction of dt past the step's start, then applies its
    update and makes the state consistent again.
    """

    stepType = "explicit"
    # (fraction of dt, stage)
    stages = ()

    def runStages(self, dt):
        for frac, stage in self.stages:
            self.titme = self.tme + frac * dt
            RHS(self)
            for blk in self:
                stage(blk, dt)
            consistify(self)

    def step(self, dt):
        self.runStages(dt)

        self.nrt += 1
        self.tme += dt
        self.titme = self.tme


class rk1(BaseExplicit):
    integratorName = "rk1"
    stages = ((0.0, ssp(0.0, 1.0, 1.0)),)


class rk2(BaseExplicit):
    integratorName = "rk2"
    stages = (
        (0.0, ssp(0.0, 1.0, 1.0, storeQ0=True)),
        (1.0, ssp(0.5, 0.5, 0.5)),
    )


class rk3(BaseExplicit):
    """
    S. Gottlieb, C.-W Shu. Total variation diminishing Runge-Kutta
    schemes. Mathematics of Computation, 67(221):73-85, 1998.
    """

    integratorName = "rk3"
    stages = (
        (0.0, ssp(0.0, 1.0, 1.0, storeQ0=True)),
        (1.0, ssp(0.75, 0.25, 0.25)),
        (0.5, ssp(1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0)),
    )


class rk34(BaseExplicit):
    integratorName = "rk34"
    stages = (
        (0.0, ssp(0.0, 1.0, 0.5, storeQ0=True)),
        (0.5, ssp(0.0, 1.0, 0.5)),
        (1.0, ssp(2.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0)),
        (0.5, ssp(0.0, 1.0, 0.5)),
    )


class maccormack(BaseExplicit):
    """Predictor then corrector, both built at the step's start time. The
    corrector's weights are rk2's second stage; only the time it is evaluated
    at differs."""

    integratorName = "maccormack"
    stages = (
        (0.0, ssp(0.0, 1.0, 1.0, storeQ0=True)),
        (0.0, ssp(0.5, 0.5, 0.5)),
    )


class rk4(BaseExplicit):
    """The classical scheme, which needs each stage's derivative kept rather
    than a running combination, so its stages are their own kernels."""

    integratorName = "rk4"
    stages = ((0.0, rk4s1), (0.5, rk4s2), (0.5, rk4s3), (1.0, rk4s4))
