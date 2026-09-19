"""The boundaries of the Euler equations: a condition an inviscid flow
can take -- a slip wall, an inlet, an exit -- with the values each reads
out of its config entry. A periodic is not one: what it does to its halo,
the exchange does as the halo lands."""

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
