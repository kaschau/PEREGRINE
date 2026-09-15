"""How a case steps in time, composed onto the solver at runtime the way
PyFR composes an integrator: getSolver picks the stepper and the controller
the config names and makes one class of the two and the solver, per case."""

from ..misc import subclassWhere
from ..multiBlock.solver import solver
from .controllers import CFL, BaseController, Fixed
from .steppers import (
    BaseStepper,
    dualTime,
    maccormack,
    rk1,
    rk2,
    rk3,
    rk34,
    rk4,
    rungeKutta,
)

__all__ = [
    "BaseController",
    "BaseStepper",
    "CFL",
    "Fixed",
    "dualTime",
    "getSolver",
    "maccormack",
    "rk1",
    "rk2",
    "rk3",
    "rk34",
    "rk4",
    "rungeKutta",
]


def getSolver(config, mesh, state=None):
    """The solver for a config: its controller and its stepper composed onto
    the base, one class per case, built when the case is."""
    ti = config["timeIntegration"]
    stepper = subclassWhere(BaseStepper, stepperName=ti["integrator"])
    controller = subclassWhere(BaseController, controllerName=ti["controller"])
    name = f"{stepper.__name__}_{controller.__name__}_solver"
    return type(name, (controller, stepper, solver), {})(config, mesh, state)
