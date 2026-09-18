"""Runge-Kutta integrators: a step is a fixed list of stages, each taking
the right-hand side, combining the state with the state the step began
from and the derivative, and making the state consistent again."""

from ..graph import Graph, LaunchNode
from .base import BaseIntegrator


class rungeKutta(BaseIntegrator):
    """A step is a fixed list of stages. Each takes the RHS, combines the
    state with the state the step began from and the derivative, and makes
    the state consistent again. The first stage keeps the state it began
    from in Q0."""

    # (fraction of dt, wQ0, wQ, wdQ) per stage: Q = wQ Q + wQ0 Q0 + wdQ dt dQ
    stages = ()

    def graphs(self):
        """Gives each stage's combination as a graph of its own, its
        weights fixed and the step read where the kernels run."""
        combine = [
            self.combineGraph(n, *weights, dt=self.dtOnDevice)
            for n, (frac, *weights) in enumerate(self.stages)
        ]
        return {**super().graphs(), "combine": combine}

    def advance(self, dt):
        solver = self.solver
        for combine in solver.graphs["combine"]:
            solver.rhs()
            combine.run()
            solver.consistify()


class rk1(rungeKutta):
    name = "rk1"
    stages = ((0.0, 0.0, 1.0, 1.0),)


class rk2(rungeKutta):
    name = "rk2"
    storage = ("Q0",)
    stages = ((0.0, 0.0, 1.0, 1.0), (1.0, 0.5, 0.5, 0.5))


class rk3(rungeKutta):
    """
    S. Gottlieb, C.-W Shu. Total variation diminishing Runge-Kutta
    schemes. Mathematics of Computation, 67(221):73-85, 1998.
    """

    name = "rk3"
    storage = ("Q0",)
    stages = (
        (0.0, 0.0, 1.0, 1.0),
        (1.0, 0.75, 0.25, 0.25),
        (0.5, 1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0),
    )


class rk34(rungeKutta):
    name = "rk34"
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

    name = "maccormack"
    storage = ("Q0",)
    stages = ((0.0, 0.0, 1.0, 1.0), (0.0, 0.5, 0.5, 0.5))


class rk4(rungeKutta):
    """The classical scheme: every stage steps from the start of the step by
    wdQ dt dQ, and wSum dt dQ of each derivative goes into a running sum,
    which is what the last stage steps by."""

    name = "rk4"
    storage = ("Q0", "Q1")
    # (fraction of dt, wdQ, wSum) per stage; the last has no wdQ of its own
    stages = (
        (0.0, 0.5, 1.0 / 6.0),
        (0.5, 0.5, 1.0 / 3.0),
        (0.5, 1.0, 1.0 / 3.0),
        (1.0, None, 1.0 / 6.0),
    )

    def combineGraph(self, n, wdQ, wSum, dt):
        Q, Q0, S, dQ, k = self.state, "Q0", "Q1", "dQ", self.kernels
        first = n == 0
        launches = []
        if first:
            launches.append(LaunchNode(self.solver.kernels["copy"], "full", A=Q0, B=Q))
        launches.append(
            LaunchNode(
                k["axpby"], "full", A=S, a=0.0 if first else 1.0, b=wSum, B=dQ, dt=dt
            )
        )
        if wdQ is None:
            # the sum already carries dt: stepped by one
            launches.append(
                LaunchNode(
                    k["axpbypcz"],
                    "full",
                    A=Q,
                    a=0.0,
                    b=1.0,
                    B=Q0,
                    c=1.0,
                    C=S,
                    dt=self.one,
                )
            )
        else:
            launches.append(
                LaunchNode(
                    k["axpbypcz"], "full", A=Q, a=0.0, b=1.0, B=Q0, c=wdQ, C=dQ, dt=dt
                )
            )
        return Graph(f"combine {n}", launches)
