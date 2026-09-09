from .base import BaseBC


class WallBC(BaseBC):
    family = "walls"


class AdiabaticNoSlipWall(WallBC):
    bcType = "adiabaticNoSlipWall"
    needsBcFam = False


class AdiabaticSlipWall(WallBC):
    bcType = "adiabaticSlipWall"
    needsBcFam = False


class AdiabaticMovingWall(WallBC):
    bcType = "adiabaticMovingWall"
    values = {"u": 1, "v": 2, "w": 3}


class IsoTNoSlipWall(WallBC):
    bcType = "isoTNoSlipWall"
    values = {"T": 4}


class IsoTSlipWall(WallBC):
    bcType = "isoTSlipWall"
    values = {"T": 4}


class IsoTMovingWall(WallBC):
    bcType = "isoTMovingWall"
    values = {"u": 1, "v": 2, "w": 3, "T": 4}
