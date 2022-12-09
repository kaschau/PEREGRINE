import itertools

import numpy as np
import pytest

from .bcBlock import create

##############################################
# Test all wall boundary conditions
##############################################

pytestmark = pytest.mark.parametrize(
    "adv,spdata",
    list(
        itertools.product(
            ("secondOrderKEEP", "fourthOrderKEEP"),
            (["Air"], "thtr_CH4_O2_Stanford_Skeletal.yaml"),
        )
    ),
)


def test_adiabaticNoSlipWall(my_setup, adv, spdata):

    mb = create("adiabaticNoSlipWall", adv, spdata)
    blk = mb[0]

    p = blk.array["q"][:, :, :, 0]
    u = blk.array["q"][:, :, :, 1]
    v = blk.array["q"][:, :, :, 2]
    w = blk.array["q"][:, :, :, 3]
    TN = blk.array["q"][:, :, :, 4::]

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
            plus = 1.0
        else:
            plus = -1.0

        face.bcFunc(blk, face, mb.eos, mb.thtrdat, "euler", mb.tme)
        blk.updateHostView(["q"])
        for s0_ in face.s0_:
            assert np.allclose(p[s0_], p[face.s1_])
            uDotn = (
                u[face.s1_] * nx * plus
                + v[face.s1_] * ny * plus
                + w[face.s1_] * nz * plus
            )

            assert np.allclose(u[s0_], u[face.s1_] - 2.0 * uDotn * plus * nx)
            assert np.allclose(v[s0_], v[face.s1_] - 2.0 * uDotn * plus * ny)
            assert np.allclose(w[s0_], w[face.s1_] - 2.0 * uDotn * plus * nz)
            assert np.allclose(TN[s0_], TN[face.s1_])

        face.bcFunc(blk, face, mb.eos, mb.thtrdat, "viscous", mb.tme)
        blk.updateHostView(["q"])
        for s0_ in face.s0_:
            assert np.allclose(u[s0_], -u[face.s1_])
            assert np.allclose(v[s0_], -v[face.s1_])
            assert np.allclose(w[s0_], -w[face.s1_])


def test_adiabaticSlipWall(my_setup, adv, spdata):

    mb = create("adiabaticSlipWall", adv, spdata)
    blk = mb[0]

    p = blk.array["q"][:, :, :, 0]
    u = blk.array["q"][:, :, :, 1]
    v = blk.array["q"][:, :, :, 2]
    w = blk.array["q"][:, :, :, 3]
    TN = blk.array["q"][:, :, :, 4::]

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
            plus = 1.0
        else:
            plus = -1.0

        face.bcFunc(blk, face, mb.eos, mb.thtrdat, "euler", mb.tme)
        blk.updateHostView(["q"])
        for s0_ in face.s0_:
            assert np.allclose(p[s0_], p[face.s1_])

            uDotn = (
                u[face.s1_] * nx * plus
                + v[face.s1_] * ny * plus
                + w[face.s1_] * nz * plus
            )

            assert np.allclose(u[s0_], u[face.s1_] - 2.0 * uDotn * plus * nx)
            assert np.allclose(v[s0_], v[face.s1_] - 2.0 * uDotn * plus * ny)
            assert np.allclose(w[s0_], w[face.s1_] - 2.0 * uDotn * plus * nz)

            assert np.allclose(TN[s0_], TN[face.s1_])


def test_adiabaticMovingWall(my_setup, adv, spdata):

    mb = create("adiabaticMovingWall", adv, spdata)
    blk = mb[0]

    p = blk.array["q"][:, :, :, 0]
    u = blk.array["q"][:, :, :, 1]
    v = blk.array["q"][:, :, :, 2]
    w = blk.array["q"][:, :, :, 3]
    TN = blk.array["q"][:, :, :, 4::]

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
            plus = 1.0
        else:
            plus = -1.0
        face.bcFunc(blk, face, mb.eos, mb.thtrdat, "euler", mb.tme)
        blk.updateHostView(["q"])
        for s0_ in face.s0_:
            assert np.allclose(p[s0_], p[face.s1_])
            uDotn = (
                u[face.s1_] * nx * plus
                + v[face.s1_] * ny * plus
                + w[face.s1_] * nz * plus
            )

            assert np.allclose(u[s0_], u[face.s1_] - 2.0 * uDotn * plus * nx)
            assert np.allclose(v[s0_], v[face.s1_] - 2.0 * uDotn * plus * ny)
            assert np.allclose(w[s0_], w[face.s1_] - 2.0 * uDotn * plus * nz)

            assert np.allclose(TN[s0_], TN[face.s1_])

        face.bcFunc(blk, face, mb.eos, mb.thtrdat, "viscous", mb.tme)
        blk.updateHostView(["q"])
        for s0_, s2_ in zip(face.s0_, face.s2_):
            assert np.allclose(
                u[s0_], 2.0 * face.array["qBcVals"][:, :, 1] - u[face.s1_]
            )
            assert np.allclose(
                v[s0_], 2.0 * face.array["qBcVals"][:, :, 2] - v[face.s1_]
            )
            assert np.allclose(
                w[s0_], 2.0 * face.array["qBcVals"][:, :, 3] - w[face.s1_]
            )


def test_isoTNoSlipWall(my_setup, adv, spdata):

    mb = create("isoTNoSlipWall", adv, spdata)
    blk = mb[0]

    p = blk.array["q"][:, :, :, 0]
    u = blk.array["q"][:, :, :, 1]
    v = blk.array["q"][:, :, :, 2]
    w = blk.array["q"][:, :, :, 3]
    T = blk.array["q"][:, :, :, 4]
    if blk.ns > 1:
        Y = blk.array["q"][:, :, :, 5::]

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
            plus = 1.0
        else:
            plus = -1.0
        face.bcFunc(blk, face, mb.eos, mb.thtrdat, "euler", mb.tme)
        blk.updateHostView(["q"])
        for s0_ in face.s0_:
            assert np.allclose(p[s0_], p[face.s1_])
            uDotn = (
                u[face.s1_] * nx * plus
                + v[face.s1_] * ny * plus
                + w[face.s1_] * nz * plus
            )

            assert np.allclose(u[s0_], u[face.s1_] - 2.0 * uDotn * plus * nx)
            assert np.allclose(v[s0_], v[face.s1_] - 2.0 * uDotn * plus * ny)
            assert np.allclose(w[s0_], w[face.s1_] - 2.0 * uDotn * plus * nz)
            assert np.allclose(T[s0_], face.array["qBcVals"][:, :, 4])
            if blk.ns > 1:
                assert np.allclose(Y[s0_], Y[face.s1_])

        face.bcFunc(blk, face, mb.eos, mb.thtrdat, "viscous", mb.tme)
        blk.updateHostView(["q"])
        for s0_ in face.s0_:
            assert np.allclose(u[s0_], -u[face.s1_])
            assert np.allclose(v[s0_], -v[face.s1_])
            assert np.allclose(w[s0_], -w[face.s1_])


def test_isoTSlipWall(my_setup, adv, spdata):

    mb = create("isoTSlipWall", adv, spdata)
    blk = mb[0]

    p = blk.array["q"][:, :, :, 0]
    u = blk.array["q"][:, :, :, 1]
    v = blk.array["q"][:, :, :, 2]
    w = blk.array["q"][:, :, :, 3]
    T = blk.array["q"][:, :, :, 4]
    if blk.ns > 1:
        Y = blk.array["q"][:, :, :, 5::]

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
            plus = 1.0
        else:
            plus = -1.0

        face.bcFunc(blk, face, mb.eos, mb.thtrdat, "euler", mb.tme)
        blk.updateHostView(["q"])
        for s0_ in face.s0_:
            assert np.allclose(p[s0_], p[face.s1_])

            uDotn = (
                u[face.s1_] * nx * plus
                + v[face.s1_] * ny * plus
                + w[face.s1_] * nz * plus
            )

            assert np.allclose(u[s0_], u[face.s1_] - 2.0 * uDotn * plus * nx)
            assert np.allclose(v[s0_], v[face.s1_] - 2.0 * uDotn * plus * ny)
            assert np.allclose(w[s0_], w[face.s1_] - 2.0 * uDotn * plus * nz)

            assert np.allclose(T[s0_], face.array["qBcVals"][:, :, 4])
            if blk.ns > 1:
                assert np.allclose(Y[s0_], Y[face.s1_])


def test_isoTMovingWall(my_setup, adv, spdata):

    mb = create("isoTMovingWall", adv, spdata)
    blk = mb[0]

    p = blk.array["q"][:, :, :, 0]
    u = blk.array["q"][:, :, :, 1]
    v = blk.array["q"][:, :, :, 2]
    w = blk.array["q"][:, :, :, 3]
    T = blk.array["q"][:, :, :, 4]
    if blk.ns > 1:
        Y = blk.array["q"][:, :, :, 5::]

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
            plus = 1.0
        else:
            plus = -1.0
        face.bcFunc(blk, face, mb.eos, mb.thtrdat, "euler", mb.tme)
        blk.updateHostView(["q"])
        for s0_ in face.s0_:
            assert np.allclose(p[s0_], p[face.s1_])

            uDotn = (
                u[face.s1_] * nx * plus
                + v[face.s1_] * ny * plus
                + w[face.s1_] * nz * plus
            )

            assert np.allclose(u[s0_], u[face.s1_] - 2.0 * uDotn * plus * nx)
            assert np.allclose(v[s0_], v[face.s1_] - 2.0 * uDotn * plus * ny)
            assert np.allclose(w[s0_], w[face.s1_] - 2.0 * uDotn * plus * nz)

            assert np.allclose(T[s0_], face.array["qBcVals"][:, :, 4])
            if blk.ns > 1:
                assert np.allclose(Y[s0_], Y[face.s1_])

        face.bcFunc(blk, face, mb.eos, mb.thtrdat, "viscous", mb.tme)
        blk.updateHostView(["q"])
        for s0_, s2_ in zip(face.s0_, face.s2_):
            assert np.allclose(
                u[s0_], 2.0 * face.array["qBcVals"][:, :, 1] - u[face.s1_]
            )
            assert np.allclose(
                v[s0_], 2.0 * face.array["qBcVals"][:, :, 2] - v[face.s1_]
            )
            assert np.allclose(
                w[s0_], 2.0 * face.array["qBcVals"][:, :, 3] - w[face.s1_]
            )
