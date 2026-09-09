from .base import BaseBC


class InteriorBC(BaseBC):
    """a face shared with another block, which the exchange fills rather than
    a boundary condition"""

    hasNeighbor = True


class Interior(InteriorBC):
    bcType = "interior"
    needsBcFam = False


class PeriodicTransLow(InteriorBC):
    bcType = "periodicTransLow"


class PeriodicTransHigh(InteriorBC):
    bcType = "periodicTransHigh"


class PeriodicRotLow(InteriorBC):
    bcType = "periodicRotLow"
    family = "periodics"


class PeriodicRotHigh(InteriorBC):
    bcType = "periodicRotHigh"
    family = "periodics"
