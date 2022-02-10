import itertools

import kokkos
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
            ("secondOrderKEEP", "fourthOrderKEEP"),
            (["Air"], "thtr_CH4_O2_Stanford_Skeletal.yaml"),
        )
    ),
)


class TestExits:
    def setup_method(self):
        kokkos.initialize()

    def teardown_method(self):
        kokkos.finalize()

    def test_constantPressureSubsonicExit(self, adv, spdata):

        mb = create("constantPressureSubsonicExit", adv, spdata)
        blk = mb[0]

        p = blk.array["q"][:, :, :, 0]
        u = blk.array["q"][:, :, :, 1]
        v = blk.array["q"][:, :, :, 2]
        w = blk.array["q"][:, :, :, 3]
        TN = blk.array["q"][:, :, :, 4::]
        for face in blk.faces:
            nface = face.nface
            print(nface)
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

            face.bcFunc(blk, face, mb.eos, mb.thtrdat, "euler")
            face.bcFunc(blk, face, mb.eos, mb.thtrdat, "viscous")

            uDotn = (u[face.s1_] * nx + v[face.s1_] * ny + w[face.s1_] * nz) * plus
            # Test for reverse flow
            outFlow = (uDotn >= 0.0,)[0]
            revFlow = (uDotn < 0.0,)[0]
            for s0_, s2_ in zip(face.s0_, face.s2_):
                # apply pressure
                assert np.allclose(p[s0_], face.array["qBcVals"][:, :, 0])

                # extrapolate everything else
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

                # extrapolate everything
                assert np.allclose(TN[s0_], 2.0 * TN[face.s1_] - TN[s2_])

            face.bcFunc(blk, face, mb.eos, mb.thtrdat, "viscous")
            for s0_ in face.s0_:
                # neumann all gradients
                assert np.allclose(blk.array["dqdx"][s0_], blk.array["dqdx"][face.s1_])
                assert np.allclose(blk.array["dqdy"][s0_], blk.array["dqdy"][face.s1_])
                assert np.allclose(blk.array["dqdz"][s0_], blk.array["dqdz"][face.s1_])

    def test_supersonicExit(self, adv, spdata):

        mb = create("supersonicExit", adv, spdata)
        blk = mb[0]

        q = blk.array["q"][:, :, :, :]
        for face in blk.faces:
            face.bcFunc(blk, face, mb.eos, mb.thtrdat, "euler")

            for s0_, s2_ in zip(face.s0_, face.s2_):
                # extrapolate everything
                assert np.allclose(q[s0_], 2.0 * q[face.s1_] - q[s2_])

            face.bcFunc(blk, face, mb.eos, mb.thtrdat, "viscous")
            for s0_ in face.s0_:
                # neumann all gradients
                assert np.allclose(blk.array["dqdx"][s0_], blk.array["dqdx"][face.s1_])
                assert np.allclose(blk.array["dqdy"][s0_], blk.array["dqdy"][face.s1_])
                assert np.allclose(blk.array["dqdz"][s0_], blk.array["dqdz"][face.s1_])
