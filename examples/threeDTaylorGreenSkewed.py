#!/usr/bin/env python
"""

Test case from

Yuichi Kuya, Soshi Kawai,
High-order accurate kinetic-energy and entropy preserving (KEEP) schemes on curvilinear grids,
Journal of Computational Physics,
Volume 442,
2021,
110482,
ISSN 0021-9991,
https://doi.org/10.1016/j.jcp.2021.110482.
(https://www.sciencedirect.com/science/article/pii/S0021999121003776)

Will reproduce test case from section 6.2

"""

from mpi4py import MPI
import kokkos
import peregrinepy as pg
import numpy as np
import matplotlib.pyplot as plt

np.seterr(all="raise")


def simulate():

    config = pg.files.configFile()
    config.validateConfig()

    mb = pg.multiBlock.generateMultiBlockSolver(1, config)
    NE = 64
    NN = 64
    NX = 64
    pg.grid.create.multiBlockCube(
        mb,
        mbDims=[1, 1, 1],
        dimsPerBlock=[NE, NN, NX],
        lengths=[2 * np.pi for _ in range(3)],
        periodic=[True, True, True],
    )

    mb.initSolverArrays(config)

    blk = mb[0]
    ng = blk.ng

    # SKEW THE GRID
    xMin = yMin = zMin = -np.pi
    Lx = Ly = Lz = 2 * np.pi
    lamXY = lamYZ = lamXZ = 8.0
    kappa = 0.25
    Ax = Ay = Az = 1.0
    delX = Lx / (NE - 1)
    delY = Ly / (NN - 1)
    delZ = Lz / (NX - 1)

    x = blk.array["x"]
    y = blk.array["y"]
    z = blk.array["z"]
    for E in range(NE):
        for N in range(NN):
            for X in range(NX):
                x[E + ng, N + ng, X + ng] = xMin + delX * (
                    E
                    + Ax
                    * np.sin(2 * np.pi * kappa)
                    * np.sin(lamXY * np.pi * N * delY / Ly)
                    * np.sin(lamXZ * np.pi * X * delZ / Lz)
                )
                y[E + ng, N + ng, X + ng] = yMin + delY * (
                    N
                    + Ay
                    * np.sin(2 * np.pi * kappa)
                    * np.sin(lamXY * np.pi * E * delX / Lx)
                    * np.sin(lamYZ * np.pi * X * delZ / Lz)
                )
                z[E + ng, N + ng, X + ng] = zMin + delZ * (
                    X
                    + Az
                    * np.sin(2 * np.pi * kappa)
                    * np.sin(lamXZ * np.pi * E * delX / Lx)
                    * np.sin(lamYZ * np.pi * N * delY / Ly)
                )

    for face in blk.faces:
        face.commRank = 0

    mb.setBlockCommunication()

    mb.unifyGrid()
    mb.computeMetrics(config["RHS"]["diffOrder"])

    R = 287.002507
    cp = 1000.0
    M0 = 0.4
    rho0 = 1.0
    gamma = cp / (cp - R)

    xc = blk.array["xc"]
    yc = blk.array["yc"]
    zc = blk.array["zc"]

    blk.array["q"][:, :, :, 0] = 1 / gamma + (rho0 * M0 ** 2 / 16.0) * (
        np.cos(2 * xc) + np.cos(2 * yc)
    ) * (np.cos(2 * zc) + 2.0)
    blk.array["q"][:, :, :, 1] = M0 * np.sin(xc) * np.cos(yc) * np.cos(zc)
    blk.array["q"][:, :, :, 2] = -M0 * np.cos(xc) * np.sin(yc) * np.cos(zc)
    blk.array["q"][:, :, :, 3] = 0.0
    blk.array["q"][:, :, :, 4] = blk.array["q"][:, :, :, 0] / (R * rho0)

    blk.updateDeviceView("q")
    mb.eos(blk, mb.thtrdat, 0, "prims")
    pg.consistify(mb)

    dt = 0.1 * 2 * np.pi / NE
    ke = []
    s = []
    t = []
    tEnd = 120 / M0
    while mb.tme < tEnd:

        if mb.nrt % 50 == 0:
            pg.misc.progressBar(mb.tme, tEnd)
            blk.updateHostView(["q", "Q"])

            rke = np.sum(
                0.5
                * blk.array["Q"][ng:-ng, ng:-ng, ng:-ng, 0]
                * (
                    blk.array["q"][ng:-ng, ng:-ng, ng:-ng, 1] ** 2
                    + blk.array["q"][ng:-ng, ng:-ng, ng:-ng, 2] ** 2
                    + blk.array["q"][ng:-ng, ng:-ng, ng:-ng, 3] ** 2
                )
                * blk.array["J"][ng:-ng, ng:-ng, ng:-ng]
            )

            rS = np.sum(
                blk.array["Q"][ng:-ng, ng:-ng, ng:-ng, 0]
                * np.log10(
                    blk.array["q"][ng:-ng, ng:-ng, ng:-ng, 0]
                    * blk.array["Q"][ng:-ng, ng:-ng, ng:-ng, 0] ** (-gamma)
                )
                * blk.array["J"][ng:-ng, ng:-ng, ng:-ng]
            )

            ke.append(rke)
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


if __name__ == "__main__":

    try:
        kokkos.initialize()
        simulate()
        kokkos.finalize()

    except Exception as e:
        import sys
        import traceback

        print(f"{e}")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        sys.exit(1)
