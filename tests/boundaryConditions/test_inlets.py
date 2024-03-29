import itertools

import numpy as np
import pytest

from .bcBlock import create

##############################################
# Test all inlet boundary conditions
##############################################

pytestmark = pytest.mark.parametrize(
    "adv,spdata",
    list(
        itertools.product(
            ("KEEPpe", "fourthOrderKEEP"),
            (["Air"], "thtr_CH4_O2_FFCMY.yaml"),
        )
    ),
)


def test_constantVelocitySubsonicInlet(my_setup, adv, spdata):
    mb = create("constantVelocitySubsonicInlet", adv, spdata)
    blk = mb[0]

    p = blk.array["q"][:, :, :, 0]
    u = blk.array["q"][:, :, :, 1]
    v = blk.array["q"][:, :, :, 2]
    w = blk.array["q"][:, :, :, 3]
    T = blk.array["q"][:, :, :, 4]
    for face in blk.faces:
        face.bcFunc(blk, face, mb.eos, mb.thtrdat, "euler", mb.tme)
        blk.updateHostView(["q"])

        for s0_, s2_ in zip(face.s0_, face.s2_):
            # extrapolate pressure
            assert np.allclose(p[s0_], 2.0 * p[face.s1_] - p[s2_])

            # apply velo on face
            assert np.allclose(u[s0_], face.array["qBcVals"][:, :, 1])
            assert np.allclose(v[s0_], face.array["qBcVals"][:, :, 2])
            assert np.allclose(w[s0_], face.array["qBcVals"][:, :, 3])

            # apply T and Ns in face
            assert np.allclose(T[s0_], face.array["qBcVals"][:, :, 4])

            if blk.ns > 1:
                for n in range(blk.ns - 1):
                    N = blk.array["q"][:, :, :, 5 + n]
                    assert np.allclose(
                        N[s0_],
                        face.array["qBcVals"][:, :, 5 + n],
                    )

        # gradients
        face.bcFunc(blk, face, mb.eos, mb.thtrdat, "postDqDxyz", mb.tme)
        blk.updateHostView(["dqdx", "dqdy", "dqdz"])

        s0_ = face.s0_[0]
        # neumann all gradients
        assert np.allclose(blk.array["dqdx"][s0_], blk.array["dqdx"][face.s1_])
        assert np.allclose(blk.array["dqdy"][s0_], blk.array["dqdy"][face.s1_])
        assert np.allclose(blk.array["dqdz"][s0_], blk.array["dqdz"][face.s1_])


def test_supersonicInlet(my_setup, adv, spdata):
    mb = create("supersonicInlet", adv, spdata)
    blk = mb[0]

    p = blk.array["q"][:, :, :, 0]
    u = blk.array["q"][:, :, :, 1]
    v = blk.array["q"][:, :, :, 2]
    w = blk.array["q"][:, :, :, 3]
    T = blk.array["q"][:, :, :, 4]
    for face in blk.faces:
        face.bcFunc(blk, face, mb.eos, mb.thtrdat, "euler", mb.tme)
        blk.updateHostView(["q"])

        for s0_, s2_ in zip(face.s0_, face.s2_):
            # extrapolate pressure
            assert np.allclose(p[s0_], face.array["qBcVals"][:, :, 0])

            # apply velo on face
            assert np.allclose(u[s0_], face.array["qBcVals"][:, :, 1])
            assert np.allclose(v[s0_], face.array["qBcVals"][:, :, 2])
            assert np.allclose(w[s0_], face.array["qBcVals"][:, :, 3])

            # apply T and Ns in face
            assert np.allclose(T[s0_], face.array["qBcVals"][:, :, 4])

            if blk.ns > 1:
                for n in range(blk.ns - 1):
                    N = blk.array["q"][:, :, :, 5 + n]
                    assert np.allclose(
                        N[s0_],
                        face.array["qBcVals"][:, :, 5 + n],
                    )

        # gradients
        face.bcFunc(blk, face, mb.eos, mb.thtrdat, "postDqDxyz", mb.tme)
        blk.updateHostView(["dqdx", "dqdy", "dqdz"])

        s0_ = face.s0_[0]
        # neumann all gradients
        assert np.allclose(blk.array["dqdx"][s0_], blk.array["dqdx"][face.s1_])
        assert np.allclose(blk.array["dqdy"][s0_], blk.array["dqdy"][face.s1_])
        assert np.allclose(blk.array["dqdz"][s0_], blk.array["dqdz"][face.s1_])


def test_constantMassFluxSubsonicInlet(my_setup, adv, spdata):
    # NOTE: fourth order not working for constant mdot
    if adv == "fourthOrderKEEP":
        return

    mb = create("constantMassFluxSubsonicInlet", adv, spdata)
    blk = mb[0]
    ng = blk.ng

    p = blk.array["q"][:, :, :, 0]
    T = blk.array["q"][:, :, :, 4]
    for face in blk.faces:
        face.bcFunc(blk, face, mb.eos, mb.thtrdat, "euler", mb.tme)
        mb.primaryAdvFlux(blk)
        blk.updateHostView(["q", "Q"])

        s1_ = face.s1_
        if face.nface in [1, 2]:
            blk.updateHostView(["iF"])
            F = blk.array["iF"][s1_][ng:-ng, ng:-ng, 0]
            S = blk.array["iS"][s1_][ng:-ng, ng:-ng]
        elif face.nface in [3, 4]:
            blk.updateHostView(["jF"])
            F = blk.array["jF"][s1_][ng:-ng, ng:-ng, 0]
            S = blk.array["jS"][s1_][ng:-ng, ng:-ng]
        elif face.nface in [5, 6]:
            blk.updateHostView(["kF"])
            F = blk.array["kF"][s1_][ng:-ng, ng:-ng, 0]
            S = blk.array["kS"][s1_][ng:-ng, ng:-ng]

        for s0_, s2_ in zip(face.s0_, face.s2_):
            # extrapolate pressure
            assert np.allclose(p[s0_], 2.0 * p[face.s1_] - p[s2_])

            # apply T and Ns in halo
            assert np.allclose(T[s0_], face.array["qBcVals"][:, :, 4])

            if blk.ns > 1:
                for n in range(blk.ns - 1):
                    N = blk.array["q"][:, :, :, 5 + n]
                    assert np.allclose(
                        N[s0_],
                        face.array["qBcVals"][:, :, 5 + n],
                    )

        # mass flux on face
        if face.nface in [2, 4, 6]:
            mult = -1.0
        else:
            mult = 1.0

        faceArea = np.sum(S)
        targetMassFlux = face.array["QBcVals"][0, 0, 0] * faceArea
        computedMassFlux = mult * np.sum(F)
        assert abs(targetMassFlux - computedMassFlux) / targetMassFlux * 100.0 < 1e-3

        # gradients
        face.bcFunc(blk, face, mb.eos, mb.thtrdat, "postDqDxyz", mb.tme)
        blk.updateHostView(["dqdx", "dqdy", "dqdz"])

        s0_ = face.s0_[0]
        # neumann all gradients
        assert np.allclose(blk.array["dqdx"][s0_], blk.array["dqdx"][face.s1_])
        assert np.allclose(blk.array["dqdy"][s0_], blk.array["dqdy"][face.s1_])
        assert np.allclose(blk.array["dqdz"][s0_], blk.array["dqdz"][face.s1_])
