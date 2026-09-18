from .topology import topology
from .grid import grid
from .restart import restart
from .solver import solver
from .haloExchange import BaseHaloExchange, DeviceHaloExchange, HostStagedHaloExchange
from ..misc import subclassWhere

__all__ = [
    "topology",
    "grid",
    "restart",
    "solver",
    "BaseHaloExchange",
    "DeviceHaloExchange",
    "HostStagedHaloExchange",
    "getHaloExchange",
]


def getHaloExchange(config):
    """Gives the halo exchange class the config names, one instance of
    which a solver makes per exchanged array."""
    return subclassWhere(BaseHaloExchange, kind=config["haloExchange"]["kind"])
