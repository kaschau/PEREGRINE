from .dualTime import dualTime
from .explicit import BaseIntegrator, maccormack, rk1, rk2, rk3, rk34, rk4

_integrators = {
    i.integratorName: i for i in (rk1, rk2, rk3, rk34, rk4, maccormack, dualTime)
}

__all__ = ["BaseIntegrator", "getIntegrator"]


def getIntegrator(ti):
    """The integrator class the config names."""
    try:
        return _integrators[ti]
    except KeyError:
        raise ValueError(f"What time integrator? {ti}")
