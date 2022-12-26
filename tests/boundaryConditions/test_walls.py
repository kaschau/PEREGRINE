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

    dpdx = blk.array["dqdx"][:, :, :, 0]
    dpdy = blk.array["dqdy"][:, :, :, 0]
    dpdz = blk.array["dqdz"][:, :, :, 0]
    dvelodx = blk.array["dqdx"][:, :, :, 1:4]
    dvelody = blk.array["dqdy"][:, :, :, 1:4]
    dvelodz = blk.array["dqdz"][:, :, :, 1:4]

    dTNdx = blk.array["dqdx"][:, :, :, 4::]
    dTNdy = blk.array["dqdy"][:, :, :, 4::]
    dTNdz = blk.array["dqdz"][:, :, :, 4::]

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

        face.bcFunc(blk, face, mb.eos, mb.thtrdat, "preDqDxyz", mb.tme)
        blk.updateHostView(["q"])
        for s0_ in face.s0_:
            assert np.allclose(u[s0_], -u[face.s1_])
            assert np.allclose(v[s0_], -v[face.s1_])
            assert np.allclose(w[s0_], -w[face.s1_])

        # Only need the first halo cell
        face.bcFunc(blk, face, mb.eos, mb.thtrdat, "postDqDxyz", mb.tme)
        blk.updateHostView(["dqdx", "dqdy", "dqdz"])
        s0_ = face.s0_[0]
        s2_ = face.s2_[0]
        # negate pressure gradients
        assert np.allclose(dpdx[s0_], -dpdx[face.s1_])
        assert np.allclose(dpdy[s0_], -dpdy[face.s1_])
        assert np.allclose(dpdz[s0_], -dpdz[face.s1_])
        # extrapolate velocity gradient
        assert np.allclose(dvelodx[s0_], 2.0 * dvelodx[face.s1_] - dvelodx[s2_])
        assert np.allclose(dvelody[s0_], 2.0 * dvelody[face.s1_] - dvelody[s2_])
        assert np.allclose(dvelodz[s0_], 2.0 * dvelodz[face.s1_] - dvelodz[s2_])
        # negate temp and species gradient (so gradient evaluates to zero on wall)
        assert np.allclose(dTNdx[s0_], -dTNdx[face.s1_])
        assert np.allclose(dTNdy[s0_], -dTNdy[face.s1_])
        assert np.allclose(dTNdz[s0_], -dTNdz[face.s1_])


def test_adiabaticSlipWall(my_setup, adv, spdata):

    mb = create("adiabaticSlipWall", adv, spdata)
    blk = mb[0]

    p = blk.array["q"][:, :, :, 0]
    u = blk.array["q"][:, :, :, 1]
    v = blk.array["q"][:, :, :, 2]
    w = blk.array["q"][:, :, :, 3]
    TN = blk.array["q"][:, :, :, 4::]

    dqdx = blk.array["dqdx"][:, :, :, :]
    dqdy = blk.array["dqdy"][:, :, :, :]
    dqdz = blk.array["dqdz"][:, :, :, :]

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

        face.bcFunc(blk, face, mb.eos, mb.thtrdat, "postDqDxyz", mb.tme)
        blk.updateHostView(["dqdx", "dqdy", "dqdz"])
        s0_ = face.s0_[0]
        # negate all gradient (so gradient evaluates to zero on wall)
        assert np.allclose(dqdx[s0_], -dqdx[face.s1_])
        assert np.allclose(dqdy[s0_], -dqdy[face.s1_])
        assert np.allclose(dqdz[s0_], -dqdz[face.s1_])


def test_adiabaticMovingWall(my_setup, adv, spdata):

    mb = create("adiabaticMovingWall", adv, spdata)
    blk = mb[0]

    p = blk.array["q"][:, :, :, 0]
    u = blk.array["q"][:, :, :, 1]
    v = blk.array["q"][:, :, :, 2]
    w = blk.array["q"][:, :, :, 3]
    TN = blk.array["q"][:, :, :, 4::]

    dvelodx = blk.array["dqdx"][:, :, :, 0:4]
    dvelody = blk.array["dqdy"][:, :, :, 0:4]
    dvelodz = blk.array["dqdz"][:, :, :, 0:4]

    dTNdx = blk.array["dqdx"][:, :, :, 4::]
    dTNdy = blk.array["dqdy"][:, :, :, 4::]
    dTNdz = blk.array["dqdz"][:, :, :, 4::]

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

        face.bcFunc(blk, face, mb.eos, mb.thtrdat, "preDqDxyz", mb.tme)
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

        face.bcFunc(blk, face, mb.eos, mb.thtrdat, "postDqDxyz", mb.tme)
        blk.updateHostView(["dqdx", "dqdy", "dqdz"])
        s0_ = face.s0_[0]
        s2_ = face.s2_[0]
        # extrapolate velocity gradient
        assert np.allclose(dvelodx[s0_], 2.0 * dvelodx[face.s1_] - dvelodx[s2_])
        assert np.allclose(dvelody[s0_], 2.0 * dvelody[face.s1_] - dvelody[s2_])
        assert np.allclose(dvelodz[s0_], 2.0 * dvelodz[face.s1_] - dvelodz[s2_])

        # negate temp and species gradient (so gradient evaluates to zero on wall)
        assert np.allclose(dTNdx[s0_], -dTNdx[face.s1_])
        assert np.allclose(dTNdy[s0_], -dTNdy[face.s1_])
        assert np.allclose(dTNdz[s0_], -dTNdz[face.s1_])


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
    dpdx = blk.array["dqdx"][:, :, :, 0]
    dpdy = blk.array["dqdy"][:, :, :, 0]
    dpdz = blk.array["dqdz"][:, :, :, 0]
    dvelodx = blk.array["dqdx"][:, :, :, 1:4]
    dvelody = blk.array["dqdy"][:, :, :, 1:4]
    dvelodz = blk.array["dqdz"][:, :, :, 1:4]

    dTdx = blk.array["dqdx"][:, :, :, 4]
    dTdy = blk.array["dqdy"][:, :, :, 4]
    dTdz = blk.array["dqdz"][:, :, :, 4]
    if blk.ns > 1:
        dYdx = blk.array["dqdx"][:, :, :, 5::]
        dYdy = blk.array["dqdy"][:, :, :, 5::]
        dYdz = blk.array["dqdz"][:, :, :, 5::]

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

        face.bcFunc(blk, face, mb.eos, mb.thtrdat, "preDqDxyz", mb.tme)
        blk.updateHostView(["q"])
        for s0_ in face.s0_:
            assert np.allclose(u[s0_], -u[face.s1_])
            assert np.allclose(v[s0_], -v[face.s1_])
            assert np.allclose(w[s0_], -w[face.s1_])

        face.bcFunc(blk, face, mb.eos, mb.thtrdat, "postDqDxyz", mb.tme)
        blk.updateHostView(["dqdx", "dqdy", "dqdz"])
        s0_ = face.s0_[0]
        s2_ = face.s2_[0]
        # negate pressure gradients
        assert np.allclose(dpdx[s0_], -dpdx[face.s1_])
        assert np.allclose(dpdy[s0_], -dpdy[face.s1_])
        assert np.allclose(dpdz[s0_], -dpdz[face.s1_])
        # extrapolate velocity gradient
        assert np.allclose(dvelodx[s0_], 2.0 * dvelodx[face.s1_] - dvelodx[s2_])
        assert np.allclose(dvelody[s0_], 2.0 * dvelody[face.s1_] - dvelody[s2_])
        assert np.allclose(dvelodz[s0_], 2.0 * dvelodz[face.s1_] - dvelodz[s2_])
        # extrap temp gradient
        assert np.allclose(dTdx[s0_], 2.0 * dTdx[face.s1_] - dTdx[s2_])
        assert np.allclose(dTdy[s0_], 2.0 * dTdy[face.s1_] - dTdy[s2_])
        assert np.allclose(dTdz[s0_], 2.0 * dTdz[face.s1_] - dTdz[s2_])
        # species gradient (so gradient evaluates to zero on wall)
        if blk.ns > 1:
            assert np.allclose(dYdx[s0_], -dYdx[face.s1_])
            assert np.allclose(dYdy[s0_], -dYdy[face.s1_])
            assert np.allclose(dYdz[s0_], -dYdz[face.s1_])


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

    dvelodx = blk.array["dqdx"][:, :, :, 0:4]
    dvelody = blk.array["dqdy"][:, :, :, 0:4]
    dvelodz = blk.array["dqdz"][:, :, :, 0:4]
    dTdx = blk.array["dqdx"][:, :, :, 4]
    dTdy = blk.array["dqdy"][:, :, :, 4]
    dTdz = blk.array["dqdz"][:, :, :, 4]
    if blk.ns > 1:
        dYdx = blk.array["dqdx"][:, :, :, 5::]
        dYdy = blk.array["dqdy"][:, :, :, 5::]
        dYdz = blk.array["dqdz"][:, :, :, 5::]

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

        face.bcFunc(blk, face, mb.eos, mb.thtrdat, "postDqDxyz", mb.tme)
        blk.updateHostView(["dqdx", "dqdy", "dqdz"])
        s0_ = face.s0_[0]
        s2_ = face.s2_[0]
        # slip wall so we neumann the velocity gradients
        assert np.allclose(dvelodx[s0_], -dvelodx[face.s1_])
        assert np.allclose(dvelody[s0_], -dvelody[face.s1_])
        assert np.allclose(dvelodz[s0_], -dvelodz[face.s1_])
        # negate temp and species gradient (so gradient evaluates to zero on wall)
        # extrap temp gradient
        assert np.allclose(dTdx[s0_], 2.0 * dTdx[face.s1_] - dTdx[s2_])
        assert np.allclose(dTdy[s0_], 2.0 * dTdy[face.s1_] - dTdy[s2_])
        assert np.allclose(dTdz[s0_], 2.0 * dTdz[face.s1_] - dTdz[s2_])
        # species gradient (so gradient evaluates to zero on wall)
        if blk.ns > 1:
            assert np.allclose(dYdx[s0_], -dYdx[face.s1_])
            assert np.allclose(dYdy[s0_], -dYdy[face.s1_])
            assert np.allclose(dYdz[s0_], -dYdz[face.s1_])


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

    dpdx = blk.array["dqdx"][:, :, :, 0]
    dpdy = blk.array["dqdy"][:, :, :, 0]
    dpdz = blk.array["dqdz"][:, :, :, 0]
    dvelodx = blk.array["dqdx"][:, :, :, 1:4]
    dvelody = blk.array["dqdy"][:, :, :, 1:4]
    dvelodz = blk.array["dqdz"][:, :, :, 1:4]
    dTdx = blk.array["dqdx"][:, :, :, 4]
    dTdy = blk.array["dqdy"][:, :, :, 4]
    dTdz = blk.array["dqdz"][:, :, :, 4]
    if blk.ns > 1:
        dYdx = blk.array["dqdx"][:, :, :, 5::]
        dYdy = blk.array["dqdy"][:, :, :, 5::]
        dYdz = blk.array["dqdz"][:, :, :, 5::]

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

        face.bcFunc(blk, face, mb.eos, mb.thtrdat, "preDqDxyz", mb.tme)
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

        face.bcFunc(blk, face, mb.eos, mb.thtrdat, "postDqDxyz", mb.tme)
        blk.updateHostView(["dqdx", "dqdy", "dqdz"])
        s0_ = face.s0_[0]
        s2_ = face.s2_[0]
        # extrapolate velocity gradient, temp
        assert np.allclose(dpdx[s0_], -dpdx[face.s1_])
        assert np.allclose(dpdy[s0_], -dpdy[face.s1_])
        assert np.allclose(dpdz[s0_], -dpdz[face.s1_])

        assert np.allclose(dvelodx[s0_], 2.0 * dvelodx[face.s1_] - dvelodx[s2_])
        assert np.allclose(dvelody[s0_], 2.0 * dvelody[face.s1_] - dvelody[s2_])
        assert np.allclose(dvelodz[s0_], 2.0 * dvelodz[face.s1_] - dvelodz[s2_])

        assert np.allclose(dTdx[s0_], 2.0 * dTdx[face.s1_] - dTdx[s2_])
        assert np.allclose(dTdy[s0_], 2.0 * dTdy[face.s1_] - dTdy[s2_])
        assert np.allclose(dTdz[s0_], 2.0 * dTdz[face.s1_] - dTdz[s2_])
        # negate species gradient (so gradient evaluates to zero on wall)
        if blk.ns > 1:
            assert np.allclose(dYdx[s0_], -dYdx[face.s1_])
            assert np.allclose(dYdy[s0_], -dYdy[face.s1_])
            assert np.allclose(dYdz[s0_], -dYdz[face.s1_])
