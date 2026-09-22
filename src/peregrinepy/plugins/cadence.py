"""How often something acts during a run."""

import numpy as np


class Cadence:
    """When something acts, from its section: after every niterOut steps,
    or at every multiple of dtOut in time; every step when the section says
    neither. A controller that can shortens a step to land on the next
    multiple; one that cannot leaves the act to the first step past it."""

    def __init__(self, cfgsect):
        self.niterOut = cfgsect.get("niterOut")
        self.dtOut = cfgsect.get("dtOut")
        if self.niterOut is None and self.dtOut is None:
            self.niterOut = 1

    def reached(self, tme):
        """The multiple of dtOut the time is at or past: on it within a few
        ulps, since a step shortened to land there is off by one addition's
        rounding."""
        tolerance = 8.0 * np.spacing(max(abs(tme), self.dtOut))
        return int((tme + tolerance) / self.dtOut)

    def next(self, solver):
        """The next time this acts at, or None when it acts by steps."""
        if self.dtOut is None:
            return None
        return (self.reached(solver.tme) + 1) * self.dtOut

    def due(self, solver):
        """Whether the step just taken is one to act on."""
        if self.dtOut is None:
            return solver.nrt % self.niterOut == 0
        tme, dt = solver.tme, solver.integrator.dt
        return self.reached(tme) != self.reached(tme - dt)

    def dueAfter(self, solver, dt):
        """Whether the step about to be taken, of :dt:, is one to act on."""
        if self.dtOut is None:
            return (solver.nrt + 1) % self.niterOut == 0
        return self.reached(solver.tme + dt) != self.reached(solver.tme)
