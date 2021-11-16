import kokkos
import peregrinepy as pg
import numpy as np
from .bcBlock import create


##############################################
# Test all exit boundary conditions
##############################################


class TestExits:
    def setup_method(self):
        kokkos.initialize()

    def teardown_method(self):
        kokkos.finalize()

    def test_constantPressureSubsonicInlet(self):

        mb = create("constantPressureSubsonicExit")
        blk = mb[0]

        q = blk.array["q"][:, :, :, 1::]
        p = blk.array["q"][:, :, :, 0]
        for face in blk.faces:
            face.bcFunc(blk, face, mb.eos, mb.thtrdat, "euler")
            face.bcFunc(blk, face, mb.eos, mb.thtrdat, "viscous")

            for s0_, s2_ in zip(face.s0_, face.s2_):
                # extrapolate everything
                assert np.allclose(q[s0_], 2.0 * q[face.s1_] - q[s2_])

                # apply pressure
                assert np.allclose(p[s0_], 2.0 * face.array["qBcVals"][0] - p[face.s1_])

            face.bcFunc(blk, face, mb.eos, mb.thtrdat, "viscous")
            for s0_ in face.s0_:
                # neumann all gradients
                assert np.allclose(blk.array["dqdx"][s0_], blk.array["dqdx"][face.s1_])
                assert np.allclose(blk.array["dqdy"][s0_], blk.array["dqdy"][face.s1_])
                assert np.allclose(blk.array["dqdz"][s0_], blk.array["dqdz"][face.s1_])

    def test_supersonicExit(self):

        mb = create("supersonicExit")
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
