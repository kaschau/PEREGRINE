import itertools

import numpy as np
import pytest

from .bcHarness import BaseBC


class Inlet(BaseBC):
    def euler(self, face):
        self.run(face, "euler")
        self.state(face)
        self.imposed(face, "T")
        self.species(face, self.imposed)


class SupersonicInlet(Inlet):
    bcType = "supersonicInlet"

    def state(self, face):
        self.imposed(face, "p")
        for c in "uvw":
            self.imposed(face, c)


class ConstantVelocitySubsonicInlet(Inlet):
    bcType = "constantVelocitySubsonicInlet"

    def state(self, face):
        self.extrapolate(face, "p")
        for c in "uvw":
            self.imposed(face, c)


class ConstantMassFluxSubsonicInlet(Inlet):
    """sets a mass flux rather than a velocity, so the check is on the flux the
    advective scheme then computes through the face"""

    bcType = "constantMassFluxSubsonicInlet"

    def state(self, face):
        self.extrapolate(face, "p")
        self._massFlux(face)

    def _massFlux(self, face):
        blk, ng = self.blk, self.blk.ng
        self.mb.primaryAdvFlux()

        d = face.direction
        F = face.boundary(getattr(blk, f"{d}F").get())[ng:-ng, ng:-ng, 0]
        S = face.boundary(blk.faceNormals(d)[0])[ng:-ng, ng:-ng]

        mult = 1.0 if face.amILow else -1.0
        target = face.QBcVals.get()[0, 0, 0] * np.sum(S)
        computed = mult * np.sum(F)
        assert abs(target - computed) / target * 100.0 < 1e-3


class StagnationSubsonicInlet(Inlet):
    """holds total conditions and lets the interior set the mass flow, so the
    halo velocity is whatever speed the isentropic solve gives -- but always
    along the face normal, which is the part that does not depend on the
    algebra"""

    bcType = "stagnationSubsonicInlet"

    def euler(self, face):
        self.run(face, "euler")
        self.alongNormal(face)
        self.species(face, self.imposed)


# the constant mass flux inlet is set aside with its post-eos fix-up
_inlets = (
    SupersonicInlet,
    ConstantVelocitySubsonicInlet,
    StagnationSubsonicInlet,
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


@pytest.mark.parametrize("bc", _inlets, ids=lambda i: i.bcType)
def test_inlet(my_setup, adv, gas, bc):
    bc(adv, gas).check()
