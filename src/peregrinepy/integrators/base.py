"""What moves a solver through time. An integrator is the scheme the
config names -- a Runge-Kutta table, or dual time around one -- holding
the solver it steps and a controller that sizes each step. It declares
what it alone needs, block arrays and kernels, and says its graphs in
names, the way the simulator does; the solver makes them."""

import numpy as np

from ..graph import Graph, LaunchNode
from ..kernel import CellCenterKernel
from ..multiBlock.arrays import CellCenterArray


class BaseIntegrator:
    """One time integration scheme: its name in the config, the block
    arrays it keeps between stages and which of them a result carries, the
    array its stages step, advance(dt), and the loop that runs the case
    with its controller sizing each step and the plugins acting."""

    # what the config calls it
    name = None
    # the block arrays it keeps between stages, (cell, ne) each
    storage = ()
    # those of them a result file carries
    restartArrays = ()
    # the block array its stages step
    state = "Q"
    # the step last taken; none before the first
    dt = None
    # set by the report plugin before a step it will report on, for a
    # scheme that has something to gather during the step
    reportDue = False

    def __init__(self, solver, controller):
        self.solver, self.controller = solver, controller
        self.config = solver.config
        # the step being taken, held where the kernels run, so a captured
        # graph reads each step's without being captured again; and a step
        # of one, for a combination whose derivative already carries the
        # step: rk4's sum, and every pseudo time stage
        self.dtOnDevice = solver.backend.allocate((1,), name="dt")
        # the step's host copy, kept: it is copied in behind the queued
        # kernels without waiting for them, so it has to stay
        self.dtHost = np.zeros(1, solver.backend.fpdtype)
        self.one = solver.backend.allocate((1,), name="one")
        self.one.set([1.0])

    ###########################################################################
    # What it needs
    ###########################################################################
    def arrays(self):
        """Gives the block arrays this integrator needs, its storage and
        its controller's: name -> what declArray takes."""
        ne = self.solver.ne
        storage = {
            name: dict(kind=CellCenterArray, components=ne) for name in self.storage
        }
        return {**storage, **self.controller.arrays()}

    def declKernels(self):
        """Makes the kernels this integrator calls, by tag -- the linear
        combinations every scheme's stages are made of, and its
        controller's -- and keeps them for its graphs."""
        self.kernels = {
            "axpby": CellCenterKernel("utils/axpby.cpp"),
            "axpbypcz": CellCenterKernel("utils/axpbypcz.cpp"),
            **self.controller.declKernels(),
        }
        return self.kernels

    def graphs(self):
        """Gives this integrator's graphs by stage, said in names."""
        return {}

    def combineGraph(self, n, wQ0, wQ, wdQ, dt):
        """Says the graph of stage :n:'s update of the state, stepping by
        :dt: -- the array holding it where the kernels run; a strong
        stability preserving combination unless a scheme says otherwise."""
        Q, dQ, k = self.state, "dQ", self.kernels
        launches = []
        if n == 0 and "Q0" in self.storage:
            launches.append(
                LaunchNode(self.solver.kernels["copy"], "full", A="Q0", B=Q)
            )
        if wQ0 == 0.0:
            launches.append(
                LaunchNode(k["axpby"], "full", A=Q, a=wQ, b=wdQ, B=dQ, dt=dt)
            )
        else:
            launches.append(
                LaunchNode(
                    k["axpbypcz"], "full", A=Q, a=wQ, b=wQ0, B="Q0", c=wdQ, C=dQ, dt=dt
                )
            )
        return Graph(f"combine {n}", launches)

    ###########################################################################
    # Stepping
    ###########################################################################
    def initialize(self):
        """Readies a fresh case for its first step; nothing, for most."""

    def restore(self, found):
        """Readies a restarted case from what the result carried of its
        restart arrays (:found:), and without what it did not; nothing, for
        most."""

    def advance(self, dt):
        """Moves the state at the solver's time to the state :dt: later."""
        raise NotImplementedError

    def step(self, dt):
        """Takes one step of :dt:: the scheme's stages, and the solver's
        clock and count move on."""
        self.dt = dt
        # queued behind the last step's kernels, ahead of this one's: no
        # wait, which would drain the device before every step
        self.dtHost[0] = dt
        self.dtOnDevice.set(self.dtHost, wait=False)
        self.advance(dt)
        self.solver.nrt += 1
        self.solver.tme += dt

    def run(self):
        """Runs the case start to finish: every step the config asks for,
        each sized by the controller, and every plugin acting as often as
        it says."""
        solver, plugins = self.solver, self.solver.plugins
        try:
            for _ in range(self.config["simulation"]["niter"]):
                dt = self.controller.stepSize()
                for plugin in plugins.values():
                    plugin.before(solver, dt)
                self.step(dt)
                for plugin in plugins.values():
                    if plugin.due(solver):
                        plugin(solver)
        finally:
            for plugin in plugins.values():
                plugin.finalize(solver)

    ###########################################################################
    # What it says
    ###########################################################################
    def stepReport(self):
        """Says what the scheme has to say about the step just taken, as
        text for the report plugin to print; nothing, for most."""

    def report(self):
        """Says how the case is integrated, for the banner."""
        return (
            f"  Time Integrator: {self.name}\n" f"  Step Size: {self.controller.name}\n"
        )
