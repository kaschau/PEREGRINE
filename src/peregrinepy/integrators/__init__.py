"""How a case steps in time: the integrator the config names, holding the
solver it moves and the controller that sizes its steps."""

from ..misc import subclassWhere
from .base import BaseIntegrator
from .controllers import CFL, BaseController, Fixed
from .dualTime import dualTime
from .rungeKutta import maccormack, rk1, rk2, rk3, rk4, rk34, rungeKutta

__all__ = [
    "BaseController",
    "BaseIntegrator",
    "CFL",
    "Fixed",
    "dualTime",
    "getIntegrator",
    "maccormack",
    "rk1",
    "rk2",
    "rk3",
    "rk34",
    "rk4",
    "rungeKutta",
]


def getIntegrator(solver):
    """Makes the integrator the solver's config names, with the controller
    its simulation section names."""
    config = solver.config
    controller = subclassWhere(BaseController, name=config["simulation"]["controller"])
    integrator = subclassWhere(
        BaseIntegrator, name=config["timeIntegration"]["integrator"]
    )
    return integrator(solver, controller(solver))
