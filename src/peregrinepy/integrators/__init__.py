from ..misc import subclassWhere
from .controllers import BaseController
from .dualTime import dualTime
from .base import BaseIntegrator
from .rungeKutta import maccormack, rk1, rk2, rk3, rk34, rk4

_integrators = {
    i.integratorName: i for i in (rk1, rk2, rk3, rk34, rk4, maccormack, dualTime)
}

__all__ = ["BaseController", "BaseIntegrator", "getController", "getIntegrator"]


def getController(cfgsect):
    """The step size controller the timeIntegration section names, built
    from it."""
    return subclassWhere(BaseController, name=cfgsect["controller"])(cfgsect)


def getIntegrator(ti):
    """The integrator class the config names."""
    try:
        return _integrators[ti]
    except KeyError:
        raise ValueError(f"What time integrator? {ti}")
