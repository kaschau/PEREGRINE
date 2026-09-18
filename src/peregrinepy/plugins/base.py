"""What a run does alongside the stepping, around the step only: a plugin
sees the solver before each step, after a step it is due on, and once more
at the end. A plugin that needs a kernel compiles and binds its own through
solver.jit; nothing a plugin needs sits on the solver. The report plugin is
everything a run prints."""


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
            dt = solver.integrator.dt
            return int(solver.tme / period) != int((solver.tme - dt) / period)
        return solver.nrt % self.everyIter == 0

    def dueAfter(self, solver, dt):
        """Whether the step about to be taken, of :dt:, is one this plugin
        acts on."""
        if self.everyTime is not None:
            period = self.everyTime
            return int((solver.tme + dt) / period) != int(solver.tme / period)
        return (solver.nrt + 1) % self.everyIter == 0

    def before(self, solver, dt):
        """Before a step of :dt:; nothing, for most."""

    def __call__(self, solver):
        raise NotImplementedError

    def finalize(self, solver):
        """When the run is over; nothing, for most."""
