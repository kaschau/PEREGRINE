"""The boundary conditions a face can carry, and putting a case onto the
faces the grid left for it.

The grid names a face and says nothing more about it, because the same grid
runs as a wall on one case and an inlet on the next. The config says what each
name is and what it reads, and this puts both on the face.
"""

from functools import cache

from ..misc import subclasses, subclassWhere
from . import exits, inlets, periodics, walls  # noqa: F401  (registers the bcs)
from .base import BaseBC

__all__ = ["BaseBC", "getBc", "validBcTypes", "conditionsOf"]


# called once per face of every block; the registry is fixed after import
@cache
def getBc(bcType):
    """The class for a bcType, which is also the check that it is one."""
    return subclassWhere(BaseBC, bcType=bcType)


@cache
def validBcTypes():
    return tuple(sorted(c.bcType for c in subclasses(BaseBC) if c.bcType))


@cache
def conditionsOf(hook):
    """Every condition with a kernel at :hook:, in the order the hook's
    kernel holds them."""
    return tuple(t for t in validBcTypes() if hook in getBc(t).hooks)
