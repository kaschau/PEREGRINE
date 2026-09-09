from .base import BaseBC


class ExitBC(BaseBC):
    family = "exits"


class ConstantPressureSubsonicExit(ExitBC):
    bcType = "constantPressureSubsonicExit"
    values = {"p": 0}


class SupersonicExit(ExitBC):
    bcType = "supersonicExit"
    needsBcFam = False
