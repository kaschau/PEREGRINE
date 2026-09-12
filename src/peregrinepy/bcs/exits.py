from .base import BaseBC


class ExitBC(BaseBC):
    family = "exits"


class ConstantPressureSubsonicExit(ExitBC):
    bcType = "constantPressureSubsonicExit"
    hooks = ("euler", "postDqDxyz")
    values = {"p": 0}


class SupersonicExit(ExitBC):
    bcType = "supersonicExit"
    hooks = ("euler", "postDqDxyz")
