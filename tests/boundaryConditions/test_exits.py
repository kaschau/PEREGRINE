import itertools

import numpy as np
import pytest

from .bcBlock import create

##############################################
# Test all exit boundary conditions
##############################################

pytestmark = pytest.mark.parametrize(
    "adv,spdata",
    list(
        itertools.product(
            ("KEEPpe", "fourthOrderKEEP"),
            (["Air"], "thtr_CH4_O2_Stanford_Skeletal.yaml"),
        )
    ),
)


def test_constantPressureSubsonicExit(my_setup, adv, spdata):
    mb = create("constantPressureSubsonicExit", adv, spdata)
    blk = mb[0]

    p = blk.array["q"][:, :, :, 0]
    u = blk.array["q"][:, :, :, 1]
    v = blk.array["q"][:, :, :, 2]
    w = blk.array["q"][:, :, :, 3]
    TN = blk.array["q"][:, :, :, 4::]
    blk.updateDeviceView("q")
    for face in blk.faces:
        nface = face.nface
        if nface in [1, 2]:
            nx = blk.array["inx"][face.s1_]
            ny = blk.array["iny"][face.s1_]
            nz = blk.array["inz"][face.s1_]
        elif nface in [3, 4]:
            nx = blk.array["jnx"][face.s1_]
            ny = blk.array["jny"][face.s1_]
            nz = blk.array["jnz"][face.s1_]
        elif nface in [5, 6]:
            nx = blk.array["knx"][face.s1_]
            ny = blk.array["kny"][face.s1_]
            nz = blk.array["knz"][face.s1_]

        if nface in [1, 3, 5]:
            plus = -1.0
        else:
            plus = 1.0

        face.bcFunc(blk, face, mb.eos, mb.thtrdat, "euler", mb.tme)
        blk.updateHostView(["q"])

        uDotn = (u[face.s1_] * nx + v[face.s1_] * ny + w[face.s1_] * nz) * plus
        # Test for reverse flow
        outFlow = (uDotn >= 0.0,)[0]
        revFlow = (uDotn < 0.0,)[0]
        for s0_, s2_ in zip(face.s0_, face.s2_):
            # apply pressure
            assert np.allclose(p[s0_], face.array["qBcVals"][:, :, 0])

            # extrapolate velocity
            assert np.allclose(
                u[s0_][outFlow], 2.0 * u[face.s1_][outFlow] - u[s2_][outFlow]
            )
            assert np.allclose(
                v[s0_][outFlow], 2.0 * v[face.s1_][outFlow] - v[s2_][outFlow]
            )
            assert np.allclose(
                w[s0_][outFlow], 2.0 * w[face.s1_][outFlow] - w[s2_][outFlow]
            )
            assert np.allclose(
                u[s0_][revFlow],
                u[face.s1_][revFlow] - 2.0 * uDotn[revFlow] * plus * nx[revFlow],
            )
            assert np.allclose(
                v[s0_][revFlow],
                v[face.s1_][revFlow] - 2.0 * uDotn[revFlow] * plus * ny[revFlow],
            )
            assert np.allclose(
                w[s0_][revFlow],
                w[face.s1_][revFlow] - 2.0 * uDotn[revFlow] * plus * nz[revFlow],
            )

            # neumann everything else
            assert np.allclose(TN[s0_], TN[face.s1_])

        # gradients
        face.bcFunc(blk, face, mb.eos, mb.thtrdat, "postDqDxyz", mb.tme)
        blk.updateHostView(["dqdx", "dqdy", "dqdz"])

        s0_ = face.s0_[0]
        # neumann all gradients
        assert np.allclose(blk.array["dqdx"][s0_], blk.array["dqdx"][face.s1_])
        assert np.allclose(blk.array["dqdy"][s0_], blk.array["dqdy"][face.s1_])
        assert np.allclose(blk.array["dqdz"][s0_], blk.array["dqdz"][face.s1_])


def test_supersonicExit(my_setup, adv, spdata):
    mb = create("supersonicExit", adv, spdata)
    blk = mb[0]

    p = blk.array["q"][:, :, :, 0]
    u = blk.array["q"][:, :, :, 1]
    v = blk.array["q"][:, :, :, 2]
    w = blk.array["q"][:, :, :, 3]
    T = blk.array["q"][:, :, :, 4]
    if blk.ns > 1:
        N = blk.array["q"][:, :, :, 5::]
    blk.updateDeviceView("q")
    for face in blk.faces:
        nface = face.nface
        if nface in [1, 2]:
            nx = blk.array["inx"][face.s1_]
            ny = blk.array["iny"][face.s1_]
            nz = blk.array["inz"][face.s1_]
        elif nface in [3, 4]:
            nx = blk.array["jnx"][face.s1_]
            ny = blk.array["jny"][face.s1_]
            nz = blk.array["jnz"][face.s1_]
        elif nface in [5, 6]:
            nx = blk.array["knx"][face.s1_]
            ny = blk.array["kny"][face.s1_]
            nz = blk.array["knz"][face.s1_]

        if nface in [1, 3, 5]:
            plus = -1.0
        else:
            plus = 1.0

        face.bcFunc(blk, face, mb.eos, mb.thtrdat, "euler", mb.tme)
        blk.updateHostView(["q"])
        uDotn = (u[face.s1_] * nx + v[face.s1_] * ny + w[face.s1_] * nz) * plus
        # Test for reverse flow
        outFlow = (uDotn >= 0.0,)[0]
        revFlow = (uDotn < 0.0,)[0]
        for s0_, s2_ in zip(face.s0_, face.s2_):
            # extrapolate pressure
            assert np.allclose(
                p[s0_], np.clip(2.0 * p[face.s1_] - p[s2_], 0.0, p[face.s1_])
            )
            # extrapolate velocity
            assert np.allclose(
                u[s0_][outFlow], 2.0 * u[face.s1_][outFlow] - u[s2_][outFlow]
            )
            assert np.allclose(
                v[s0_][outFlow], 2.0 * v[face.s1_][outFlow] - v[s2_][outFlow]
            )
            assert np.allclose(
                w[s0_][outFlow], 2.0 * w[face.s1_][outFlow] - w[s2_][outFlow]
            )
            assert np.allclose(
                u[s0_][revFlow],
                u[face.s1_][revFlow] - 2.0 * uDotn[revFlow] * plus * nx[revFlow],
            )
            assert np.allclose(
                v[s0_][revFlow],
                v[face.s1_][revFlow] - 2.0 * uDotn[revFlow] * plus * ny[revFlow],
            )
            assert np.allclose(
                w[s0_][revFlow],
                w[face.s1_][revFlow] - 2.0 * uDotn[revFlow] * plus * nz[revFlow],
            )
            # extrapolate TN
            assert np.allclose(T[s0_], np.clip(2.0 * T[face.s1_] - T[s2_], 0.0, None))
            if blk.ns > 1:
                assert np.allclose(
                    N[s0_], np.clip(2.0 * N[face.s1_] - N[s2_], 0.0, 1.0)
                )

        # gradients
        face.bcFunc(blk, face, mb.eos, mb.thtrdat, "postDqDxyz", mb.tme)
        blk.updateHostView(["dqdx", "dqdy", "dqdz"])

        s0_ = face.s0_[0]
        # neumann all gradients
        assert np.allclose(blk.array["dqdx"][s0_], blk.array["dqdx"][face.s1_])
        assert np.allclose(blk.array["dqdy"][s0_], blk.array["dqdy"][face.s1_])
        assert np.allclose(blk.array["dqdz"][s0_], blk.array["dqdz"][face.s1_])
