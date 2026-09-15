from .base import BaseBC


class Interior(BaseBC):
    """a face shared with another block, which the exchange fills rather than
    a boundary condition"""

    bcType = "interior"
    hasNeighbor = True


class PeriodicTrans(Interior):
    """shared with a block it has been moved onto without being turned: the
    vectors in its halo are unchanged, so it is an interior face"""

    bcType = "periodicTrans"


class PeriodicRot(BaseBC):
    """shared with a block it has been turned onto, so every vector in its
    halo turns with it; how far is a property of the grid, not the case"""

    bcType = "periodicRot"
    hasNeighbor = True
    family = "periodics"
