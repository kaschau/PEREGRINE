from .base import BaseIntegrator


class RKIntegrator(BaseIntegrator):
    """A step is a fixed list of stages. Each takes the RHS at a fraction of
    dt past the step's start, combines the state with the state the step
    began from and the derivative, and makes the state consistent again.
    The first stage keeps the state it began from in Q0."""

    stepType = "explicit"
    sources = ("utils/axpby.cpp", "utils/axpbypcz.cpp")
    # the array the stages step
    state = "Q"
    # (fraction of dt, wQ0, wQ, wdQ) per stage: Q = wQ Q + wQ0 Q0 + wdQ dt dQ
    stages = ()

    def step(self, tme, dt, report=False):
        for n, (frac, *weights) in enumerate(self.stages):
            self.rhs.run(tme + frac * dt)
            self.combine(*weights, dt=dt, first=n == 0)
            self.consistify.run(tme + frac * dt)

    def combine(self, wQ0, wQ, wdQ, dt, first):
        """One stage's update of the state; a strong stability preserving
        combination unless a scheme says otherwise."""
        Q, dQ = self.state, "dQ"
        if first and self.nStorage:
            self.axpby(A="Q0", a=0.0, b=1.0, B=Q)
        if wQ0 == 0.0:
            self.axpby(A=Q, a=wQ, b=wdQ * dt, B=dQ)
        else:
            self.axpbypcz(A=Q, a=wQ, b=wQ0, B="Q0", c=wdQ * dt, C=dQ)


class rk1(RKIntegrator):
    integratorName = "rk1"
    stages = ((0.0, 0.0, 1.0, 1.0),)


class rk2(RKIntegrator):
    integratorName = "rk2"
    nStorage = 1
    stages = ((0.0, 0.0, 1.0, 1.0), (1.0, 0.5, 0.5, 0.5))


class rk3(RKIntegrator):
    """
    S. Gottlieb, C.-W Shu. Total variation diminishing Runge-Kutta
    schemes. Mathematics of Computation, 67(221):73-85, 1998.
    """

    integratorName = "rk3"
    nStorage = 1
    stages = (
        (0.0, 0.0, 1.0, 1.0),
        (1.0, 0.75, 0.25, 0.25),
        (0.5, 1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0),
    )


class rk34(RKIntegrator):
    integratorName = "rk34"
    nStorage = 1
    stages = (
        (0.0, 0.0, 1.0, 0.5),
        (0.5, 0.0, 1.0, 0.5),
        (1.0, 2.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0),
        (0.5, 0.0, 1.0, 0.5),
    )


class maccormack(RKIntegrator):
    """Predictor then corrector, both built at the step's start time. The
    corrector's weights are rk2's second stage; only the time it is evaluated
    at differs."""

    integratorName = "maccormack"
    nStorage = 1
    stages = ((0.0, 0.0, 1.0, 1.0), (0.0, 0.5, 0.5, 0.5))


class rk4(RKIntegrator):
    """The classical scheme: every stage steps from the start of the step by
    wdQ dt dQ, and wSum dt dQ of each derivative goes into a running sum,
    which is what the last stage steps by."""

    integratorName = "rk4"
    nStorage = 2
    # (fraction of dt, wdQ, wSum) per stage; the last has no wdQ of its own
    stages = (
        (0.0, 0.5, 1.0 / 6.0),
        (0.5, 0.5, 1.0 / 3.0),
        (0.5, 1.0, 1.0 / 3.0),
        (1.0, None, 1.0 / 6.0),
    )

    def combine(self, wdQ, wSum, dt, first):
        Q, Q0, S, dQ = "Q", "Q0", "Q1", "dQ"
        if first:
            self.axpby(A=Q0, a=0.0, b=1.0, B=Q)
        self.axpby(A=S, a=0.0 if first else 1.0, b=wSum * dt, B=dQ)
        if wdQ is None:
            self.axpbypcz(A=Q, a=0.0, b=1.0, B=Q0, c=1.0, C=S)
        else:
            self.axpbypcz(A=Q, a=0.0, b=1.0, B=Q0, c=wdQ * dt, C=dQ)
