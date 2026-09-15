import itertools

import pytest

from .bcHarness import BaseBC


class Wall(BaseBC):
    # pressure and species always negate; the other two rules are the two axes
    # a Wall varies on
    _tempGrad = None
    _veloGrad = None

    @property
    def gradRules(self):
        return dict(p="negate", velo=self._veloGrad, T=self._tempGrad, Y="negate")

    def euler(self, face):
        self.run(face, "euler")
        self.mirror(face, "p")
        self.reflect(face)
        self.temperature(face)
        if self.blk.ns > 1:
            self.mirror(face, "Y")


class Adiabatic(Wall):
    """no heat flux, so the temperature gradient averages to zero on the wall"""

    _tempGrad = "negate"

    def temperature(self, face):
        self.mirror(face, "T")


class IsoT(Wall):
    """the wall temperature is held, so its gradient is whatever it needs to be"""

    _tempGrad = "neumann"

    def temperature(self, face):
        self.imposed(face, "T")


class Slip(Wall):
    _veloGrad = "negate"


class NoSlip(Wall):
    _veloGrad = "neumann"

    def viscous(self, face):
        self.run(face, "preDqDxyz")
        for c in "uvw":
            self.negate(face, c)


class Moving(Wall):
    _veloGrad = "neumann"

    def viscous(self, face):
        self.run(face, "preDqDxyz")
        for c in "uvw":
            self.straddles(face, c)


class AdiabaticSlipWall(Slip, Adiabatic):
    bcType = "adiabaticSlipWall"


class AdiabaticNoSlipWall(NoSlip, Adiabatic):
    bcType = "adiabaticNoSlipWall"


class AdiabaticMovingWall(Moving, Adiabatic):
    bcType = "adiabaticMovingWall"


class IsoTSlipWall(Slip, IsoT):
    bcType = "isoTSlipWall"


class IsoTNoSlipWall(NoSlip, IsoT):
    bcType = "isoTNoSlipWall"


class IsoTMovingWall(Moving, IsoT):
    bcType = "isoTMovingWall"


_walls = (
    AdiabaticSlipWall,
    AdiabaticNoSlipWall,
    AdiabaticMovingWall,
    IsoTSlipWall,
    IsoTNoSlipWall,
    IsoTMovingWall,
)

pytestmark = pytest.mark.parametrize(
    "adv,gas",
    list(
        itertools.product(
            ("KEPaEC", "fourthOrderKEEP"),
            ("air", "CH4_O2"),
        )
    ),
)


@pytest.mark.parametrize("bc", _walls, ids=lambda w: w.bcType)
def test_wall(my_setup, adv, gas, bc):
    bc(adv, gas).check()
