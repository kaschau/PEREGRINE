"""The boundaries of the Navier-Stokes equations: every one Euler has, with
gradient bodies, and the walls the flow sticks to."""

from ..boundaries import BaseBC, InletBC, MassFluxInletBC


class BaseNSBC(BaseBC):
    """A boundary of the Navier-Stokes equations: bodies at the euler hook
    and the gradient hooks."""


class AdiabaticSlipWall(BaseNSBC):
    bcType = "adiabaticSlipWall"


class IsoTSlipWall(BaseNSBC):
    bcType = "isoTSlipWall"
    values = {"T": 4}


class ConstantVelocitySubsonicInlet(InletBC, BaseNSBC):
    bcType = "constantVelocitySubsonicInlet"
    values = {"u": 1, "v": 2, "w": 3, "T": 4}


class SupersonicInlet(InletBC, BaseNSBC):
    bcType = "supersonicInlet"
    values = {"p": 0, "u": 1, "v": 2, "w": 3, "T": 4}


class StagnationSubsonicInlet(InletBC, BaseNSBC):
    bcType = "stagnationSubsonicInlet"
    values = {"pt": 0, "Tt": 4}


class ConstantMassFluxSubsonicInlet(MassFluxInletBC, BaseNSBC):
    bcType = "constantMassFluxSubsonicInlet"
    values = {"T": 4}


class ConstantPressureSubsonicExit(BaseNSBC):
    bcType = "constantPressureSubsonicExit"
    values = {"p": 0}


class SupersonicExit(BaseNSBC):
    bcType = "supersonicExit"


class AdiabaticNoSlipWall(BaseNSBC):
    bcType = "adiabaticNoSlipWall"


class AdiabaticMovingWall(BaseNSBC):
    bcType = "adiabaticMovingWall"
    values = {"u": 1, "v": 2, "w": 3}


class IsoTNoSlipWall(BaseNSBC):
    bcType = "isoTNoSlipWall"
    values = {"T": 4}


class IsoTMovingWall(BaseNSBC):
    bcType = "isoTMovingWall"
    values = {"u": 1, "v": 2, "w": 3, "T": 4}
