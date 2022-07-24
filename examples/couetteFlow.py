#!/usr/bin/env python
"""

Couette Flow with top wall moving at 5m/s

"""

from mpi4py import MPI
import kokkos
import peregrinepy as pg
import numpy as np
import matplotlib.pyplot as plt

wallSpeed = 5.0
ny = 10
h = 0.025

n = np.array([i for i in range(50)][1::])


def analytical(y, h, t, nu):
    return wallSpeed * y / h - 2 * wallSpeed / np.pi * np.sum(
        1.0
        / n
        * np.exp(-(n ** 2) * np.pi ** 2 * nu * t / h ** 2)
        * np.sin(n * np.pi * (1 - y / h))
    )


def simulate():

    config = pg.files.configFile()
    config["timeIntegration"]["dt"] = 10 * 2.0e-5 / ny
    config["RHS"]["diffusion"] = True
    config["thermochem"]["trans"] = "constantProps"
    config["thermochem"]["spdata"] = ["Air"]

    mb = pg.multiBlock.generateMultiBlockSolver(1, config)
    pg.grid.create.multiBlockCube(
        mb,
        mbDims=[1, 1, 1],
        dimsPerBlock=[2, ny, 2],
        lengths=[0.01, h, 0.01],
        periodic=[True, False, False],
    )

    mb.initSolverArrays(config)

    blk = mb[0]

    # face 1
    blk.getFace(1).commRank = 0
    # face 2
    blk.getFace(2).commRank = 0
    # face 3
    blk.getFace(3).bcType = "adiabaticNoSlipWall"
    # face 4 isoT moving wall
    blk.getFace(4).bcType = "adiabaticMovingWall"

    for face in [5, 6]:
        blk.getFace(face).bcType = "adiabaticSlipWall"

    valueDict = {"u": wallSpeed, "v": 0.0, "w": 0.0}
    face4 = blk.getFace(4)
    face4.array["qBcVals"] = np.zeros(blk.array["q"][face4.s1_].shape)
    pg.bcs.prepWalls.prep_adiabaticMovingWall(blk, face4, valueDict)
    pg.misc.createViewMirrorArray(face4, "qBcVals", blk.array["q"][face4.s1_].shape)

    mb.setBlockCommunication()

    mb.unifyGrid()
    mb.computeMetrics(config["RHS"]["diffOrder"])

    ng = blk.ng
    blk.array["q"][ng:-ng, ng:-ng, ng, 0] = 101325.0
    blk.array["q"][:, :, :, 1:4] = 0.0
    blk.array["q"][ng:-ng, ng:-ng, ng, 4] = 300.0

    blk.updateDeviceView(["q"])
    mb.eos(blk, mb.thtrdat, 0, "prims")
    pg.consistify(mb)

    mu = np.unique(mb.thtrdat.array["mu0"])[0]
    blk.updateHostView(["Q"])
    rho = np.unique(blk.array["Q"][:, :, :, 0])[0]
    nu = mu / rho

    outputTimes = [0.0005, 0.005, 0.05]

    doneOutput = [False, False, False]
    outputU = []
    simTme = max(outputTimes) * h ** 2 / nu
    while mb.tme < simTme:
        mb.step(config["timeIntegration"]["dt"])

        if mb.nrt % 200 == 0:
            pg.misc.progressBar(mb.tme, simTme)
            blk.updateHostView(["Q"])
            if np.any(np.isnan(blk.array["Q"])):
                raise ValueError("Nan detected")

        for i, oT in enumerate(outputTimes):
            t = oT * h ** 2 / nu
            if mb.tme >= t and not doneOutput[i]:
                blk.updateHostView(["q"])
                outputU.append(blk.array["q"][ng, ng:-ng, ng, 1].copy())
                doneOutput[i] = True

    # Analytical solution
    yplot = np.linspace(0, h, 100)
    anSol = []
    for oT in outputTimes:
        sol = []
        for yy in yplot:
            t = oT * h ** 2 / nu
            sol.append(analytical(yy, h, t, nu))
        anSol.append(np.array(sol))

    fig, ax1 = plt.subplots()
    ax1.grid(True, linestyle="--")
    ax1.set_title("Couette Results")
    ax1.set_xlabel(r"$u/U$")
    ax1.set_ylabel(r"$y/h$")
    y = blk.array["yc"][ng, ng:-ng, ng] / h
    y = np.append(y, [1.0])
    legends = [str(i) for i in outputTimes]
    for oU, oA, legend in zip(outputU, anSol, legends):
        ax1.scatter(np.append(oU, [wallSpeed]) / wallSpeed, y, label=legend, s=15.0)
        ax1.plot(oA / wallSpeed, yplot / h, linewidth=0.5, color="k")
    ax1.scatter(
        np.linspace(0, 1, y.shape[0]),
        y,
        marker="o",
        facecolor="None",
        edgecolor="b",
        label="Steady State",
        linewidth=0.5,
    )
    ax1.legend(title=r"$h^2t/\nu$")
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
