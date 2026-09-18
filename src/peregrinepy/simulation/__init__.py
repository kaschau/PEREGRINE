"""The physics a solver can simulate, each a package: the spec of what it
needs and the graphs it runs, and its boundaries; the config's
`simulation.physics` names one."""

from ..misc import subclassWhere
from .base import BaseSimulation
from .boundaries import BaseBC
from .euler import BaseEulerBC, EulerSimulation
from .navierStokes import BaseNSBC, NavierStokesSimulation

__all__ = [
    "BaseBC",
    "BaseEulerBC",
    "BaseNSBC",
    "BaseSimulation",
    "EulerSimulation",
    "NavierStokesSimulation",
    "getSimulation",
]


def getSimulation(config):
    """Makes the simulation the config names, validated."""
    physics = config["simulation"]["physics"]
    return subclassWhere(BaseSimulation, physics=physics)(config)
