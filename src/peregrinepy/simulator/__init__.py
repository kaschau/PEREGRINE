"""The physics a solver can simulate, each a package: the spec of what it
needs and the graphs it runs, and its boundaries; the config's
`simulation.physics` names one."""

from ..misc import subclassWhere
from .base import BaseSimulator
from .boundaries import BaseBC
from .euler import BaseEulerBC, EulerSimulator
from .navierStokes import BaseNSBC, NavierStokesSimulator

__all__ = [
    "BaseBC",
    "BaseEulerBC",
    "BaseNSBC",
    "BaseSimulator",
    "EulerSimulator",
    "NavierStokesSimulator",
    "getSimulator",
]


def getSimulator(config):
    """Makes the simulator the config names, validated."""
    name = config["simulation"]["simulator"]
    return subclassWhere(BaseSimulator, name=name)(config)
