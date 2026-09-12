from .base import BaseBC


class WallBC(BaseBC):
    family = "walls"


class AdiabaticNoSlipWall(WallBC):
    bcType = "adiabaticNoSlipWall"
    hooks = ("euler", "preDqDxyz", "postDqDxyz")


class AdiabaticSlipWall(WallBC):
    bcType = "adiabaticSlipWall"
    hooks = ("euler", "postDqDxyz")


class AdiabaticMovingWall(WallBC):
    bcType = "adiabaticMovingWall"
    hooks = ("euler", "preDqDxyz", "postDqDxyz")
    values = {"u": 1, "v": 2, "w": 3}


class IsoTNoSlipWall(WallBC):
    bcType = "isoTNoSlipWall"
    hooks = ("euler", "preDqDxyz", "postDqDxyz")
    values = {"T": 4}


class IsoTSlipWall(WallBC):
    bcType = "isoTSlipWall"
    hooks = ("euler", "postDqDxyz")
    values = {"T": 4}


class IsoTMovingWall(WallBC):
    bcType = "isoTMovingWall"
    hooks = ("euler", "preDqDxyz", "postDqDxyz")
    values = {"u": 1, "v": 2, "w": 3, "T": 4}
