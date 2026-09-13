class BasePlugin:
    """Something a run does alongside the stepping, as often as its config
    section says: every so many steps, or every so much time."""

    # what the config calls it
    name = None

    def __init__(self, solver, cfgsect):
        self.everyIter = cfgsect.get("everyIter")
        self.everyTime = cfgsect.get("everyTime")
        if self.everyIter is None and self.everyTime is None:
            self.everyIter = 1

    def due(self, solver):
        """Whether the step just taken is one this plugin acts on."""
        if self.everyTime is not None:
            period = self.everyTime
            return int(solver.tme / period) != int((solver.tme - solver.dt) / period)
        return solver.nrt % self.everyIter == 0

    def __call__(self, solver):
        raise NotImplementedError

    def finalize(self, solver):
        """When the run is over; nothing, for most."""
