import peregrinepy as pg
import numpy as np
from .bcBlock import create


##############################################
# Test all inlet boundary conditions
##############################################


def test_constantVelocitySubsonicInlet():

    import kokkos

    kokkos.initialize()

    mb = create("constantVelocitySubsonicInlet")
    blk = mb[0]

    p = blk.array["q"][:, :, :, 0]
    u = blk.array["q"][:, :, :, 1]
    v = blk.array["q"][:, :, :, 2]
    w = blk.array["q"][:, :, :, 3]
    T = blk.array["q"][:, :, :, 4]
    for face in blk.faces:
        face.bcFunc(blk, face, mb.eos, mb.thtrdat, "euler")

        for s0_, s2_ in zip(face.s0_, face.s2_):
            # extrapolate pressure
            assert np.allclose(p[s0_], 2.0 * p[face.s1_] - p[s2_])

            # apply velo on face
            assert np.allclose(u[s0_], 2.0 * face.array["qBcVals"][1] - u[face.s1_])
            assert np.allclose(v[s0_], 2.0 * face.array["qBcVals"][2] - v[face.s1_])
            assert np.allclose(w[s0_], 2.0 * face.array["qBcVals"][3] - w[face.s1_])

            # apply T and Ns in face
            assert np.allclose(T[s0_], 2.0 * face.array["qBcVals"][4] - T[face.s1_])

            if blk.ns > 1:
                for n in range(blk.ns - 1):
                    N = blk.array["q"][:, :, :, 5 + n]
                    assert np.allclose(
                        N[s0_], 2.0 * face.array["qBcVals"][5 + n] - N[face.s1_]
                    )

        face.bcFunc(blk, face, mb.eos, mb.thtrdat, "viscous")
        for s0_ in face.s0_:
            # neumann all gradients
            assert np.allclose(blk.array["dqdx"][s0_], blk.array["dqdx"][face.s1_])
            assert np.allclose(blk.array["dqdy"][s0_], blk.array["dqdy"][face.s1_])
            assert np.allclose(blk.array["dqdz"][s0_], blk.array["dqdz"][face.s1_])

    kokkos.finalize()


def test_supersonicInlet():

    import kokkos

    kokkos.initialize()

    mb = create("supersonicInlet")
    blk = mb[0]

    p = blk.array["q"][:, :, :, 0]
    u = blk.array["q"][:, :, :, 1]
    v = blk.array["q"][:, :, :, 2]
    w = blk.array["q"][:, :, :, 3]
    T = blk.array["q"][:, :, :, 4]
    for face in blk.faces:
        face.bcFunc(blk, face, mb.eos, mb.thtrdat, "euler")

        for s0_, s2_ in zip(face.s0_, face.s2_):
            # extrapolate pressure
            assert np.allclose(p[s0_], 2.0 * face.array["qBcVals"][0] - p[face.s1_])

            # apply velo on face
            assert np.allclose(u[s0_], 2.0 * face.array["qBcVals"][1] - u[face.s1_])
            assert np.allclose(v[s0_], 2.0 * face.array["qBcVals"][2] - v[face.s1_])
            assert np.allclose(w[s0_], 2.0 * face.array["qBcVals"][3] - w[face.s1_])

            # apply T and Ns in face
            assert np.allclose(T[s0_], 2.0 * face.array["qBcVals"][4] - T[face.s1_])

            if blk.ns > 1:
                for n in range(blk.ns - 1):
                    N = blk.array["q"][:, :, :, 5 + n]
                    assert np.allclose(
                        N[s0_], 2.0 * face.array["qBcVals"][5 + n] - N[face.s1_]
                    )

        face.bcFunc(blk, face, mb.eos, mb.thtrdat, "viscous")
        for s0_ in face.s0_:
            # neumann all gradients
            assert np.allclose(blk.array["dqdx"][s0_], blk.array["dqdx"][face.s1_])
            assert np.allclose(blk.array["dqdy"][s0_], blk.array["dqdy"][face.s1_])
            assert np.allclose(blk.array["dqdz"][s0_], blk.array["dqdz"][face.s1_])

    kokkos.finalize()
