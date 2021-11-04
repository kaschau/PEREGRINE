#!/usr/bin/env python
"""

Test case from

High-order accurate kinetic-energy and entropy preserving (KEEP) schemes on curvilinear grids
https://doi.org/10.1016/j.jcp.2021.110482

Should reproduce figure 3 depending on resolution setting.


"""

from mpi4py import MPI
import kokkos
import peregrinepy as pg
import numpy as np
import matplotlib.pyplot as plt

np.seterr(all="raise")


def simulate():

    config = pg.files.configFile()

    mb = pg.multiBlock.generateMultiBlockSolver(1, config)
    NE = NN = 41
    pg.grid.create.multiBlockCube(
        mb,
        mbDims=[1, 1, 1],
        dimsPerBlock=[NE, NN, 2],
        lengths=[1, 1, 0.01],
    )

    mb.initSolverArrays(config)

    blk = mb[0]
    ng = blk.ng

    # SKEW THE GRID
    x0 = y0 = 0.0
    Lx = Ly = 12.0
    delX = Lx / (NE - 1)
    delY = Ly / (NN - 1)
    Ax = Ay = 1.0
    xMin = yMin = -6.0
    lamX = lamY = 4.0
    kappa = 0.25
    x = blk.array["x"]
    y = blk.array["y"]
    for E in range(NE):
        for N in range(NN):
            x[E + ng, N + ng, :] = xMin + delX * (
                E
                + Ax * np.sin(2 * np.pi * kappa) * np.sin(lamX * np.pi * N * delY / Ly)
            )
            y[E + ng, N + ng, :] = yMin + delY * (
                N
                + Ay * np.sin(2 * np.pi * kappa) * np.sin(lamY * np.pi * E * delX / Lx)
            )

    for face in blk.faces:
        face.bcType = "b1"
        face.bcFam = None
        face.neighbor = 0
        face.orientation = "123"
        face.commRank = 0
    for f in [5, 6]:
        face = blk.getFace(f)
        face.bcType = "adiabaticSlipWall"
        face.bcFam = None
        face.neighbor = None
        face.orientation = "000"
        face.commRank = None

    pg.mpiComm.blockComm.setBlockCommunication(mb)

    mb.unifyGrid()
    mb.computeMetrics()

    Rc = 1.0
    rhoInf = 1.0
    MInf = 0.1
    pInf = 101325.0
    R = 281.4583333333333
    cp = 1000.0
    cv = cp - R
    gamma = cp / cv
    TInf = pInf / (R * rhoInf)
    aInf = np.sqrt(gamma * R * TInf)
    uInf = MInf * aInf
    C0 = 0.02 * uInf * Rc

    xc = blk.array["xc"]
    yc = blk.array["yc"]

    r = np.sqrt(((xc - x0) ** 2 + (yc - y0) ** 2) / Rc ** 2)

    # u
    blk.array["q"][:, :, :, 1] = uInf - (C0 * (yc - y0) / Rc ** 2) * np.exp(
        -(r ** 2) / 2.0
    )
    # v
    blk.array["q"][:, :, :, 2] = (C0 * (xc - x0) / Rc ** 2) * np.exp(-(r ** 2) / 2.0)

    # p
    blk.array["q"][:, :, :, 0] = pInf - rhoInf * C0 ** 2 / (2.0 * Rc ** 2) * np.exp(
        -(r ** 2) / 2.0
    )
    # T
    blk.array["q"][:, :, :, 4] = blk.array["q"][:, :, :, 0] / (R * rhoInf)

    mb.eos(blk, mb.thtrdat, 0, "prims")
    pg.consistify(mb)

    refX = xc[ng:-ng, int(NN / 2.0), ng] / Rc
    refV = np.copy(blk.array["q"][ng:-ng, int(NN / 2.0), ng, 2] / uInf)

    # pg.writers.writeGrid(mb, config["io"]["griddir"])
    # pg.writers.writeRestart(mb, config["io"]["outputdir"], gridPath="../Grid")

    dt = 0.1 * (Lx / NE) / aInf
    tEnd = Lx / uInf
    while mb.tme < tEnd:
        if mb.nrt % 50 == 0:
            pg.misc.progressBar(mb.tme, tEnd)
        mb.step(dt)

    # plot v/Uinf
    plt.plot(
        refX, blk.array["q"][ng:-ng, int(NN / 2.0), ng, 2] / uInf, label=f"{NE = }"
    )
    plt.plot(refX, refV, "o", label="exact")
    plt.ylim([-0.016, 0.016])
    plt.xlim([-6, 6])
    plt.title("2D Euler Vortex Results")
    plt.legend()
    plt.show()
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
