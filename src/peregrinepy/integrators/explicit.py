from ..kernel import BoundKernel


def cppStage(name):
    """A compute kernel used as a stage of its own."""

    def stage(ti, dt):
        getattr(ti, name)(dt=dt)

    return stage


def ssp(wQ0, wQ, wdQ, storeQ0=False):
    """A stage of a strong stability preserving scheme,
    Q = wQ Q + wQ0 Q0 + wdQ dt dQ. The stage that begins a step keeps the
    state it began from, since later stages combine with it."""

    def stage(ti, dt):
        views = ti.table.views
        if storeQ0:
            ti.axpby(A=views("Q0"), a=0.0, b=1.0, B=views("Q"))
        if wQ0 == 0.0:
            ti.axpby(A=views("Q"), a=wQ, b=wdQ * dt, B=views("dQ"))
        else:
            ti.axpbypcz(
                A=views("Q"), a=wQ, b=wQ0, B=views("Q0"), c=wdQ * dt, C=views("dQ")
            )

    return stage


class BaseIntegrator:
    """How a case steps in time: from the state at one time to the state a
    step later, through the case's graphs and the kernels of its own stages.
    It is built with what it steps -- the block table, the species data, the
    graphs -- and told the time it steps from."""

    integratorName = None
    stepType = None
    # how many Q registers the stages combine through
    nStorage = 0
    # the kernels the stages call
    sources = ()

    def __init__(self, table, thtrdat, graphs, config):
        self.table = table
        self.rhs = graphs["rhs"]
        self.consistify = graphs["consistify"]
        self.consistifyFromPrims = graphs["consistifyFromPrims"]
        self.kernels = [BoundKernel(table, thtrdat, source) for source in self.sources]
        for kernel in self.kernels:
            setattr(self, kernel.__name__, kernel)

    def initialize(self):
        """What a fresh case does before its first step; nothing, for most."""

    def restore(self, blocks, nrt, path):
        """What a restarted case reads back beyond its state; nothing, for most."""

    def writeState(self, blocks, nrt, path):
        """What a result holds beyond the state; nothing, for most."""

    def step(self, tme, dt, report=False):
        """The state at :tme: to the state at :tme: + :dt:."""
        raise NotImplementedError


class BaseExplicit(BaseIntegrator):
    """A step is a fixed list of stages. Each one sets the time the RHS is
    built at, as a fraction of dt past the step's start, then applies its
    update and makes the state consistent again.
    """

    stepType = "explicit"
    # (fraction of dt, stage)
    stages = ()
    sources = ("utils/axpby.cpp", "utils/axpbypcz.cpp")

    def step(self, tme, dt, report=False):
        for frac, stage in self.stages:
            self.rhs.run(tme + frac * dt)
            stage(self, dt)
            self.consistify.run(tme + frac * dt)


class rk1(BaseExplicit):
    integratorName = "rk1"
    nStorage = 0
    stages = ((0.0, ssp(0.0, 1.0, 1.0)),)


class rk2(BaseExplicit):
    integratorName = "rk2"
    nStorage = 1
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
    nStorage = 1
    stages = (
        (0.0, ssp(0.0, 1.0, 1.0, storeQ0=True)),
        (1.0, ssp(0.75, 0.25, 0.25)),
        (0.5, ssp(1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0)),
    )


class rk34(BaseExplicit):
    integratorName = "rk34"
    nStorage = 1
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
    nStorage = 1
    stages = (
        (0.0, ssp(0.0, 1.0, 1.0, storeQ0=True)),
        (0.0, ssp(0.5, 0.5, 0.5)),
    )


class rk4(BaseExplicit):
    """The classical scheme, which needs each stage's derivative kept rather
    than a running combination, so its stages are their own kernels."""

    integratorName = "rk4"
    nStorage = 4
    sources = BaseExplicit.sources + tuple(
        f"timeIntegration/rk4s{n}.cpp" for n in (1, 2, 3, 4)
    )
    stages = tuple(
        (frac, cppStage(kernel))
        for frac, kernel in (
            (0.0, "rk4s1"),
            (0.5, "rk4s2"),
            (0.5, "rk4s3"),
            (1.0, "rk4s4"),
        )
    )
