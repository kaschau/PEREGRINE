from .base import BaseBC


class InteriorBC(BaseBC):
    """a face shared with another block, which the exchange fills rather than
    a boundary condition. A periodic is one of these that has been moved, and
    how far it moves is a property of the grid, so none of them read a case."""

    hasNeighbor = True


class Interior(InteriorBC):
    bcType = "interior"


class PeriodicTrans(InteriorBC):
    """moved without being turned, so the vectors in its halo are unchanged
    and there is nothing for a kernel to do"""

    bcType = "periodicTrans"


class PeriodicRot(InteriorBC):
    """turned onto its partner, so every vector in its halo turns with it"""

    bcType = "periodicRot"
    family = "periodics"
