#!/usr/bin/env python
"""

Couette Flow with top wall moving at 5m/s

"""
import kokkos
import peregrinepy as pg
import numpy as np
import matplotlib.pyplot as plt

np.seterr(all="raise")


def simulate():

    wallSpeed = 5.0

    config = pg.files.configFile()
    config["simulation"]["dt"] = 2.0e-5
    config["simulation"]["niter"] = 50000
    config["RHS"]["diffusion"] = True

    ny = 10
    mb = pg.multiBlock.generateMultiBlockSolver(1, config)
    pg.grid.create.multiBlockCube(
        mb,
        mbDims=[1, 1, 1],
        dimsPerBlock=[2, ny, 2],
        lengths=[0.01, 0.025, 0.01],
    )

    mb.initSolverArrays(config)

    blk = mb[0]

    # face 1
    blk.getFace(1).bcFam = None
    blk.getFace(1).bcType = "b1"
    blk.getFace(1).neighbor = 0
    blk.getFace(1).orientation = "123"
    blk.getFace(1).commRank = 0
    # face 2
    blk.getFace(2).bcFam = None
    blk.getFace(2).bcType = "b1"
    blk.getFace(2).neighbor = 0
    blk.getFace(2).orientation = "123"
    blk.getFace(2).commRank = 0
    # face 3
    blk.getFace(3).bcFam = None
    blk.getFace(3).bcType = "adiabaticNoSlipWall"
    blk.getFace(3).neighbor = None
    blk.getFace(3).orientation = None
    blk.getFace(3).commRank = 0
    # face 4 isoT moving wall
    blk.getFace(4).bcFam = "whoosh"
    blk.getFace(4).bcType = "adiabaticMovingWall"
    blk.getFace(4).neighbor = None
    blk.getFace(4).orientation = None
    blk.getFace(4).commRank = 0

    for face in [5, 6]:
        blk.getFace(face).bcFam = None
        blk.getFace(face).bcType = "adiabaticSlipWall"
        blk.getFace(face).commRank = 0

    blk.getFace(4).bcVals = {"u": wallSpeed, "v": 0.0, "w": 0.0}

    pg.mpiComm.blockComm.setBlockCommunication(mb)

    mb.unifyGrid()
    mb.computeMetrics()

    ng = blk.ng
    blk.array["q"][ng:-ng, ng:-ng, ng, 0] = 101325.0
    blk.array["q"][:, :, :, 1:4] = 0.0
    blk.array["q"][ng:-ng, ng:-ng, ng, 4] = 300.0

    mb.eos(blk, mb.thtrdat, 0, "prims")
    pg.consistify(mb)

    for niter in range(config["simulation"]["niter"]):
        mb.step(config["simulation"]["dt"])

        if mb.nrt % 200 == 0:
            pg.misc.progressBar(mb.nrt, config["simulation"]["niter"])

    fig, ax1 = plt.subplots()
    ax1.set_title("Couette Results")
    ax1.set_xlabel(r"x")
    y = blk.array["yc"][ng, ng:-ng, ng]
    u = blk.array["q"][ng, ng:-ng, ng, 1]
    ax1.plot(u, y, color="k", label="u", linewidth=0.5)
    ax1.scatter(
        np.linspace(0, wallSpeed, y.shape[0]),
        y,
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
