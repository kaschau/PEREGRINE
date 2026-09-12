#!/usr/bin/env python
"""

Test case from

Preventing spurious pressure oscillations in split convective form discretization for compressible flows
https://doi.org/10.1016/j.jcp.2020.110060

Should reproduce results in Fig. 2 for the KEEP scheme (blue line)


"""

from mpi4py import MPI  # noqa: F401

import peregrinepy as pg
import numpy as np
import matplotlib.pyplot as plt

# the ideal gas the case assumes, stated in full
air = {"Air": {"MW": 28.97, "cp0": 1000.0}}


np.seterr(all="raise")


def simulate():
    config = pg.files.configFile()
    config["mcPhysics"]["mixture"] = air
    config.validateConfig()

    mb = pg.multiBlock.buildSolver(config, 1)
    NE = 65
    NN = 65
    NX = 65
    pg.mesher.CubeMesher(
        mbDims=[1, 1, 1],
        dimsPerBlock=[NE, NN, NX],
        lengths=[2 * np.pi for _ in range(3)],
        periodic=[True, True, True],
    ).mesh(mb)

    blk = mb[0]
    ng = blk.ng

    for face in blk.faces:
        face.commRank = 0

    mb.setBlockCommunication()

    mb.unifyGrid()
    mb.computeMetrics()

    R = 287.002507
    cp = 1000.0
    cv = cp - R
    M0 = 0.4
    rho0 = 1.0
    gamma = cp / (cp - R)
    xc, yc, zc = (blk.hostCopy("cells")[..., n] for n in range(3))
    q = blk.q.get()
    q[:, :, :, 0] = 1 / gamma + (rho0 * M0**2 / 16.0) * (
        np.cos(2 * xc) + np.cos(2 * yc)
    ) * (np.cos(2 * zc + 2.0))
    q[:, :, :, 1] = M0 * np.sin(xc) * np.cos(yc) * np.cos(zc)
    q[:, :, :, 2] = -M0 * np.cos(xc) * np.sin(yc) * np.cos(zc)
    q[:, :, :, 4] = q[:, :, :, 0] / (R * rho0)

    blk.q.set(q)
    mb.eos(blk, mb.thtrdat, 0, "prims")
    pg.consistify(mb)

    dt = 0.1 * 2 * np.pi / 64
    ke = []
    e = []
    s = []
    t = []
    tEnd = 120.0 / M0
    while mb.tme < tEnd:
        if mb.nrt % 50 == 0:
            pg.misc.progressBar(mb.tme, tEnd)
            q, Q = blk.q.get(), blk.Q.get()

            rke = np.sum(
                0.5
                * Q[ng:-ng, ng:-ng, ng:-ng, 0]
                * (
                    q[ng:-ng, ng:-ng, ng:-ng, 1] ** 2
                    + q[ng:-ng, ng:-ng, ng:-ng, 2] ** 2
                    + q[ng:-ng, ng:-ng, ng:-ng, 3] ** 2
                )
            )
            re = np.sum(
                Q[ng:-ng, ng:-ng, ng:-ng, 0] * (q[ng:-ng, ng:-ng, ng:-ng, 4] * cv)
            )

            rS = np.sum(
                Q[ng:-ng, ng:-ng, ng:-ng, 0]
                * np.log10(
                    q[ng:-ng, ng:-ng, ng:-ng, 0]
                    * Q[ng:-ng, ng:-ng, ng:-ng, 0] ** (-gamma)
                )
            )

            ke.append(rke)
            e.append(re)
            s.append(rS)
            t.append(mb.tme * M0)

        mb.step(dt)

    plt.plot(t, ke / ke[0])
    plt.ylim([0, 2.4])
    plt.title(r"$\rho k / (\rho k)_{0}$")
    plt.savefig("ke.png")
    plt.clf()
    plt.plot(t, (-(s - s[0])) / s[0])
    plt.ylim([-3e-2, 1e-2])
    plt.title(r"$\Delta(\rho s) / (\rho_0 s_0)$")
    plt.savefig("entropy.png")
    plt.clf()
    plt.plot(t, e / e[0])
    plt.savefig("e.png")
    plt.clf()


if __name__ == "__main__":
    try:
        pg.abi.initialize()
        simulate()
        pg.abi.finalize()

    except Exception as e:
        import sys
        import traceback

        print(f"{e}")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        sys.exit(1)
