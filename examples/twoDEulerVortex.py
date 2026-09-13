#!/usr/bin/env python
"""

Test case from

High-order accurate kinetic-energy and entropy preserving (KEEP) schemes on curvilinear grids
https://doi.org/10.1016/j.jcp.2021.110482

Should reproduce figure 3 depending on resolution setting.


"""

from mpi4py import MPI  # noqa: F401
import peregrinepy as pg
import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt

# the ideal gas the case assumes, stated in full
air = {"Air": {"MW": 28.97, "cp0": 1000.0}}


np.seterr(all="raise")


def simulate():
    config = pg.files.configFile()
    config["mcPhysics"]["mixture"] = air
    config.validateConfig()

    NE = NN = 41
    mb = pg.multiBlock.solver(
        config,
        mesh=pg.mesher.CubeMesher(
            mbDims=[1, 1, 1],
            dimsPerBlock=[NE, NN, 2],
            lengths=[12, 12, 0.01],
            periodic=[True, True, False],
        ),
    )

    blk = mb.blocks[0]
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
    nodes = blk.hostCopy("nodes")
    x, y = (nodes[..., n] for n in range(2))
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
    blk.store("nodes", nodes)
    # the nodes moved, so the halos and metrics follow them
    mb.unifyGrid()
    mb.computeMetrics()

    for f in [5, 6]:
        face = blk.getFace(f)
        face.bcType = "adiabaticSlipWall"

    print(mb)

    Rc = 1.0
    rhoInf = 1.0
    MInf = 0.1
    pInf = 101325.0
    R = 287.002507
    cp = 1000.0
    cv = cp - R
    gamma = cp / cv
    TInf = pInf / (R * rhoInf)
    aInf = np.sqrt(gamma * R * TInf)
    uInf = MInf * aInf
    C0 = 0.02 * uInf * Rc

    xc, yc = (blk.hostCopy("cells")[..., n] for n in range(2))

    r = np.sqrt(((xc - x0) ** 2 + (yc - y0) ** 2) / Rc**2)

    q = blk.q.get()
    # u
    q[:, :, :, 1] = uInf - (C0 * (yc - y0) / Rc**2) * np.exp(-(r**2) / 2.0)
    # v
    q[:, :, :, 2] = (C0 * (xc - x0) / Rc**2) * np.exp(-(r**2) / 2.0)

    # p
    q[:, :, :, 0] = pInf - rhoInf * C0**2 / (2.0 * Rc**2) * np.exp(-(r**2) / 2.0)
    # T
    q[:, :, :, 4] = q[:, :, :, 0] / (R * rhoInf)

    blk.q.set(q)
    mb.stateFromPrims(nface=0)
    mb.consistify()

    refX = xc[ng:-ng, int(NN / 2.0), ng] / Rc
    refV = np.copy(q[ng:-ng, int(NN / 2.0), ng, 2] / uInf)

    dt = 0.1 * (Lx / NE) / aInf
    tEnd = Lx / uInf

    ts = perf_counter()
    bar = pg.misc.Progress(tEnd)
    while mb.tme < tEnd:
        if mb.nrt % 50 == 0:
            bar.at(mb.tme)
        mb.step(dt)
    print(f"Time integration took {perf_counter()-ts} seconds.")

    q = blk.q.get()
    # plot v/Uinf
    plt.plot(refX, q[ng:-ng, int(NN / 2.0), ng, 2] / uInf, label=f"{NE = }")
    plt.plot(refX, refV, "o", label="exact")
    plt.ylim([-0.016, 0.016])
    plt.xlim([-6, 6])
    plt.title("2D Euler Vortex Results")
    plt.legend()
    plt.show()
    plt.clf()


if __name__ == "__main__":
    try:
        pg.abi.lib.initialize()
        simulate()
        pg.abi.lib.finalize()

    except Exception as e:
        import sys
        import traceback

        print(f"{e}")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        sys.exit(1)
