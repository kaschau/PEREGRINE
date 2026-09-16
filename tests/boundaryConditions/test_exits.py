import itertools

import pytest

from .bcHarness import BaseBC


class Exit(BaseBC):
    def _flowMasks(self, face):
        """where the flow leaves the block, and where it re-enters"""
        n, sign = self.normals(face)
        velo = [self.q(c) for c in "uvw"]
        uDotn = sum(face.interior(c)[0] * ni for c, ni in zip(velo, n)) * sign
        return uDotn >= 0.0, uDotn < 0.0

    def euler(self, face):
        self.run(face, "euler")
        out, rev = self._flowMasks(face)
        self.state(face)
        for c in "uvw":
            self.extrapolate(face, c, where=out)
        # reflected on the way in, so reverse flow cannot drag the halo inside
        self.reflect(face, where=rev)


class ConstantPressureSubsonicExit(Exit):
    """the exit pressure is imposed, everything else rides along"""

    bcType = "constantPressureSubsonicExit"

    def state(self, face):
        self.imposed(face, "p")
        self.mirror(face, "T")
        if self.blk.ns > 1:
            self.mirror(face, "Y")


class SupersonicExit(Exit):
    """nothing is imposed: extrapolate, but never past physical bounds"""

    bcType = "supersonicExit"

    def state(self, face):
        # p and T floor at a hundredth of the interior, so the halo keeps a
        # density
        p, T = face.interior(self.q("p"))[0], face.interior(self.q("T"))[0]
        self.extrapolate(face, "p", lo=0.01 * p, hi=p)
        self.extrapolate(face, "T", lo=0.01 * T)
        if self.blk.ns > 1:
            self.extrapolate(face, "Y", lo=0.0, hi=1.0)


_exits = (ConstantPressureSubsonicExit, SupersonicExit)

pytestmark = pytest.mark.parametrize(
    "adv,gas",
    list(
        itertools.product(
            ("KEPaEC",),
            ("air", "CH4_O2"),
        )
    ),
)


@pytest.mark.parametrize("bc", _exits, ids=lambda e: e.bcType)
def test_exit(my_setup, adv, gas, bc):
    case = bc(adv, gas)
    case.check()
