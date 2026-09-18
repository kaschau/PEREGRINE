"""What a run does alongside the stepping: a plugin sees the solver before
each step, after a step it is due on, and once more at the end; and one
may add kernels to the end of a stage's graphs, a sponge on the viscosity
after consistify. A plugin is made from its config section before the
solver is built, so it can say what arrays and kernels it needs like the
simulation does, and starts on the solver once it is. The report plugin is
everything a run prints."""


class BasePlugin:
    """Something a run does alongside the stepping, as often as its config
    section says: every so many steps, or every so much time."""

    # what the config calls it
    name = None

    def __init__(self, cfgsect):
        self.everyIter = cfgsect.get("everyIter")
        self.everyTime = cfgsect.get("everyTime")
        if self.everyIter is None and self.everyTime is None:
            self.everyIter = 1

    def arrays(self):
        """Gives the arrays this plugin needs on every block, name ->
        declArray's keywords; none, for most."""
        return {}

    def declKernels(self):
        """Makes the kernels this plugin launches, by tag, compiled with the
        solver's; none, for most."""
        return {}

    def after(self):
        """Gives the nodes this plugin adds to the end of a stage's graphs,
        stage -> [nodes]; none, for most."""
        return {}

    def start(self, solver):
        """Takes what it needs of the built solver; nothing, for most."""

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
        """After a step it is due on; nothing, for one that only adds to
        the graphs."""

    def finalize(self, solver):
        """When the run is over; nothing, for most."""
