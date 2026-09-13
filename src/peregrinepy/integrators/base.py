from ..kernel import BoundKernel


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
    # the block arrays it keeps beyond the state, which a result carries too
    stateArrays = ()

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

    def restore(self, found):
        """What a restarted case does with what the result carried of its
        state arrays (:found:), and without what it did not; nothing, for
        most."""

    def step(self, tme, dt, report=False):
        """The state at :tme: to the state at :tme: + :dt:."""
        raise NotImplementedError
