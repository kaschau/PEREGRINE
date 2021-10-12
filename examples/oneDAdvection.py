#!/usr/bin/env python
"""

Test case from

Preventing spurious pressure oscillations in split convective form discretization for compressible flows
https://doi.org/10.1016/j.jcp.2020.110060

Should reproduce results in Fig. 1 for the KEEP scheme (blue line)


"""

import kokkos
import peregrinepy as pg
import numpy as np
import matplotlib.pyplot as plt


def simulate():

    config = pg.files.configFile()
    config["RHS"]["diffusion"] = False
    mb = pg.multiBlock.generateMultiBlockSolver(1, config)
    pg.grid.create.multiBlockCube(
        mb,
        mbDims=[1, 1, 1],
        dimsPerBlock=[41, 2, 2],
        lengths=[1, 0.01, 0.01],
    )
    mb.initSolverArrays(config)

    blk = mb[0]
    for face in blk.faces:
        face.bcType = "adiabaticSlipWall"

    blk.getFace(1).bcType = "b1"
    blk.getFace(1).neighbor = 0
    blk.getFace(1).orientation = "123"
    blk.getFace(1).commRank = 0

    blk.getFace(2).bcType = "b1"
    blk.getFace(2).neighbor = 0
    blk.getFace(2).orientation = "123"
    blk.getFace(2).commRank = 0

    pg.mpiComm.blockComm.setBlockCommunication(mb)

    mb.unifyGrid()

    mb.computeMetrics()

    ng = blk.ng
    R = 281.4583333333333
    blk.array["q"][:, :, :, 0] = 1.0
    blk.array["q"][:, :, :, 1] = 1.0
    initial_rho = 2.0 + np.sin(2 * np.pi * blk.array["xc"][ng:-ng, 0, 0])
    initial_T = 1.0 / (R * initial_rho)
    blk.array["q"][ng:-ng, ng, ng, 4] = initial_T

    # Update cons
    mb.eos(blk, mb.thtrdat, 0, "prims")
    pg.consistify(mb)

    dt = 0.1 * 0.025
    tEnd = 11.0
    while mb.tme < tEnd:

        if mb.nrt % 50 == 0:
            pg.misc.progressBar(mb.tme, tEnd)

        mb.step(dt)

    # Compute primatives from conserved Q
    fig, ax1 = plt.subplots()
    ax1.set_title("1D Advection Results")
    ax1.set_xlabel(r"x")
    x = blk.array["xc"][ng:-ng, ng, ng]
    rho = blk.array["Q"][ng:-ng, ng, ng, 0]
    p = blk.array["q"][ng:-ng, ng, ng, 0]
    u = blk.array["q"][ng:-ng, ng, ng, ng]
    ax1.plot(x, rho, color="g", label="rho", linewidth=0.5)
    ax1.plot(x, p, color="r", label="p", linewidth=0.5)
    ax1.plot(x, u, color="k", label="u", linewidth=0.5)
    ax1.scatter(
        x,
        initial_rho,
        marker="o",
        facecolor="w",
        edgecolor="b",
        label="exact",
        linewidth=0.5,
    )
    ax1.legend()
    plt.show()
    plt.close()


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
