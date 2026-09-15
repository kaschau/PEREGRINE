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

__all__ = ["BaseBC", "getBc", "validBcTypes", "bcTypesWith"]


# called once per face of every block; the registry is fixed after import
@cache
def getBc(bcType):
    """The class for a bcType, which is also the check that it is one."""
    return subclassWhere(BaseBC, bcType=bcType)


@cache
def validBcTypes():
    return tuple(sorted(c.bcType for c in subclasses(BaseBC) if c.bcType))


@cache
def bcTypesWith(bcHook):
    """Every bcType with a kernel at :bcHook:, in sorted order, which is the
    order they launch in."""
    return tuple(t for t in validBcTypes() if bcHook in getBc(t).bcHooks())
