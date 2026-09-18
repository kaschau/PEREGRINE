"""The boundaries of the Euler equations: a condition an inviscid flow
can take -- a slip wall, an inlet, an exit -- with the values each reads
out of its config entry, and the turned periodic, whose body turns what
the exchange brought."""

from ..boundaries import BaseBC, InletBC, MassFluxInletBC


class BaseEulerBC(BaseBC):
    """A boundary of the Euler equations: it has a body at the euler
    hook."""


class AdiabaticSlipWall(BaseEulerBC):
    bcType = "adiabaticSlipWall"


class IsoTSlipWall(BaseEulerBC):
    bcType = "isoTSlipWall"
    values = {"T": 4}


class ConstantVelocitySubsonicInlet(InletBC, BaseEulerBC):
    bcType = "constantVelocitySubsonicInlet"
    values = {"u": 1, "v": 2, "w": 3, "T": 4}


class SupersonicInlet(InletBC, BaseEulerBC):
    bcType = "supersonicInlet"
    values = {"p": 0, "u": 1, "v": 2, "w": 3, "T": 4}


class StagnationSubsonicInlet(InletBC, BaseEulerBC):
    bcType = "stagnationSubsonicInlet"
    values = {"pt": 0, "Tt": 4}


class ConstantMassFluxSubsonicInlet(MassFluxInletBC, BaseEulerBC):
    bcType = "constantMassFluxSubsonicInlet"
    values = {"T": 4}


class ConstantPressureSubsonicExit(BaseEulerBC):
    bcType = "constantPressureSubsonicExit"
    values = {"p": 0}


class SupersonicExit(BaseEulerBC):
    bcType = "supersonicExit"


class PeriodicRot(BaseEulerBC):
    """Meets a block turned onto it, so its body turns every vector in the
    halo after the exchange; how far is the grid's to say, not the
    case's. The only boundary with a neighbor that has a body: an interior
    or translated periodic face is the exchange alone."""

    bcType = "periodicRot"
