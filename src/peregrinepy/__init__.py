# For pytests to work, we cannot not initialize here
# since each test needs to initialize and finalize each
# time. Also for scripting, we dont need MPI.
#
# So for running real cases, we initialize in runPeregrine
# which I think makes more sense anyway.
import mpi4py.rc

mpi4py.rc.initialize = False

from . import bcs  # noqa: E402
from . import compute  # noqa: E402
from . import coproc  # noqa: E402
from . import files  # noqa: E402
from . import grid  # noqa: E402
from . import interpolation  # noqa: E402
from . import misc  # noqa: E402
from . import mpiComm  # noqa: E402
from . import multiBlock  # noqa: E402
from . import readers  # noqa: E402
from . import thermoTransport  # noqa: E402
from . import writers  # noqa: E402
from ._version import __version__  # noqa: E402
from .bootstrapCase import bootstrapCase  # noqa: E402
from .consistify import consistify  # noqa: E402
from .RHS import RHS  # noqa: E402

__all__ = [
    "bcs",
    "compute",
    "coproc",
    "files",
    "grid",
    "interpolation",
    "misc",
    "mpiComm",
    "multiBlock",
    "readers",
    "thermoTransport",
    "writers",
    "__version__",
    "bootstrapCase",
    "consistify",
    "RHS",
]
