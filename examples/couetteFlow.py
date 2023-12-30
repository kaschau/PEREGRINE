#!/usr/bin/env python
"""

Couette Flow with top wall moving at 5m/s

"""

from mpi4py import MPI

import peregrinepy as pg
import numpy as np
import matplotlib.pyplot as plt


def analytical(y, h, t, nu, wallSpeed):
    n = np.array([i for i in range(50)][1::])
    mult = wallSpeed / abs(wallSpeed)
    return mult * (
        wallSpeed * y / h
        - 2
        * wallSpeed
        / np.pi
        * np.sum(
            1.0
            / n
            * np.exp(-(n**2) * np.pi**2 * nu * t / h**2)
            * np.sin(n * np.pi * (1 - y / h))
        )
    )


def simulate(index, velo):
    wallSpeed = 5.0
    nx = 50
    h = 0.025
    if "-" in velo:
        wallSpeed *= -1.0

    if index == "i":
        assert ("y" in velo) or ("z" in velo)
    elif index == "j":
        assert ("x" in velo) or ("z" in velo)
    elif index == "k":
        assert ("y" in velo) or ("x" in velo)

    config = pg.files.configFile()
    config["timeIntegration"]["dt"] = 10 * 1.0e-5 / nx
    config["RHS"]["diffusion"] = True
    config["thermochem"]["trans"] = "constantProps"
    config["thermochem"]["spdata"] = ["Air"]
    config.validateConfig()

    mb = pg.multiBlock.generateMultiBlockSolver(1, config)
    rot = {"i": 0, "j": 1, "k": 2}

    def rotate(li, index):
        return li[-rot[index] :] + li[: -rot[index]]

    dimsPerBlock = rotate([nx, 2, 2], index)
    lengths = rotate([h, 0.001, 0.001], index)

    if "x" in velo:
        periodic = [True, False, False]
    elif "y" in velo:
        periodic = [False, True, False]
    elif "z" in velo:
        periodic = [False, False, True]

    pg.grid.create.multiBlockCube(
        mb,
        mbDims=[1, 1, 1],
        dimsPerBlock=dimsPerBlock,
        lengths=lengths,
        periodic=periodic,
    )

    mb.initSolverArrays(config)

    blk = mb[0]

    if index == "i":
        blk.getFace(1).bcType = "adiabaticNoSlipWall"
        blk.getFace(2).bcType = "adiabaticMovingWall"
        if "y" in velo:
            blk.getFace(3).commRank = 0
            blk.getFace(4).commRank = 0
            for face in [5, 6]:
                blk.getFace(face).bcType = "adiabaticSlipWall"
        else:
            blk.getFace(5).commRank = 0
            blk.getFace(6).commRank = 0
            for face in [3, 4]:
                blk.getFace(face).bcType = "adiabaticSlipWall"
    elif index == "j":
        blk.getFace(3).bcType = "adiabaticNoSlipWall"
        blk.getFace(4).bcType = "adiabaticMovingWall"
        if "x" in velo:
            blk.getFace(1).commRank = 0
            blk.getFace(2).commRank = 0
            for face in [5, 6]:
                blk.getFace(face).bcType = "adiabaticSlipWall"
        else:
            blk.getFace(5).commRank = 0
            blk.getFace(6).commRank = 0
            for face in [1, 2]:
                blk.getFace(face).bcType = "adiabaticSlipWall"
    elif index == "k":
        blk.getFace(5).bcType = "adiabaticNoSlipWall"
        blk.getFace(6).bcType = "adiabaticMovingWall"
        if "x" in velo:
            blk.getFace(1).commRank = 0
            blk.getFace(2).commRank = 0
            for face in [3, 4]:
                blk.getFace(face).bcType = "adiabaticSlipWall"
        else:
            blk.getFace(3).commRank = 0
            blk.getFace(4).commRank = 0
            for face in [1, 2]:
                blk.getFace(face).bcType = "adiabaticSlipWall"
    else:
        raise ValueError()

    if "x" in velo:
        valueDict = {"u": wallSpeed, "v": 0.0, "w": 0.0}
    elif "y" in velo:
        valueDict = {"u": 0.0, "v": wallSpeed, "w": 0.0}
    elif "z" in velo:
        valueDict = {"u": 0.0, "v": 0.0, "w": wallSpeed}
    else:
        raise ValueError()
    for face in blk.faces:
        if face.bcType == "adiabaticMovingWall":
            face.array["qBcVals"] = np.zeros(blk.array["q"][face.s1_].shape)
            pg.bcs.prepWalls.prep_adiabaticMovingWall(blk, face, valueDict)
            pg.misc.createViewMirrorArray(
                face, "qBcVals", blk.array["q"][face.s1_].shape
            )
            break

    mb.setBlockCommunication()

    mb.unifyGrid()
    mb.computeMetrics()

    ng = blk.ng
    blk.array["q"][:, :, :, 0] = 101325.0
    blk.array["q"][:, :, :, 1:4] = 0.0
    blk.array["q"][:, :, :, 4] = 300.0

    blk.updateDeviceView(["q"])
    mb.eos(blk, mb.thtrdat, 0, "prims")
    pg.consistify(mb)

    mu = np.unique(mb.thtrdat.array["mu0"])[0]
    blk.updateHostView(["Q"])
    rho = np.unique(blk.array["Q"][:, :, :, 0])[0]
    nu = mu / rho

    if index == "i":
        s_ = np.s_[ng:-ng, ng, ng]
    elif index == "j":
        s_ = np.s_[ng, ng:-ng, ng]
    elif index == "k":
        s_ = np.s_[ng, ng, ng:-ng]
    ccArray = {"i": "xc", "j": "yc", "k": "zc"}
    if "x" in velo:
        uIndex = 1
    elif "y" in velo:
        uIndex = 2
    elif "z" in velo:
        uIndex = 3
    else:
        raise ValueError()

    xc = blk.array[ccArray[index]][s_]
    sU_ = s_ + (uIndex,)

    outputTimes = [0.0005, 0.005, 0.05]
    doneOutput = [False for _ in range(len(outputTimes))]
    outputU = []
    simTme = max(outputTimes) * h**2 / nu
    while mb.tme < simTme:
        mb.step(config["timeIntegration"]["dt"])

        if mb.nrt % 200 == 0:
            pg.misc.progressBar(mb.tme, simTme)
            blk.updateHostView(["Q"])
            if np.any(np.isnan(blk.array["Q"])):
                raise ValueError("Nan detected")

        for i, oT in enumerate(outputTimes):
            t = oT * h**2 / nu
            if mb.tme >= t and not doneOutput[i]:
                blk.updateHostView(["q"])
                outputU.append(blk.array["q"][sU_].copy())
                doneOutput[i] = True

    # Analytical solution
    yplot = np.linspace(0, h, 100)
    anSol = []
    for oT in outputTimes:
        sol = []
        for yy in yplot:
            t = oT * h**2 / nu
            sol.append(analytical(yy, h, t, nu, wallSpeed))
        anSol.append(np.array(sol))

    fig, ax1 = plt.subplots()
    ax1.grid(True, linestyle="--")
    ax1.set_title("Couette Results")
    ax1.set_xlabel(r"$u/U$")
    ax1.set_ylabel(r"$y/h$")
    y = xc / h
    y = np.append(y, [1.0])
    legends = [str(i) for i in outputTimes]
    for oU, oA, legend in zip(outputU, anSol, legends):
        ax1.scatter(
            np.append(oU, [wallSpeed]) / abs(wallSpeed), y, label=legend, s=15.0
        )
        ax1.plot(oA / wallSpeed, yplot / h, linewidth=0.5, color="k")
    ax1.scatter(
        np.linspace(0, wallSpeed / abs(wallSpeed), y.shape[0]),
        np.linspace(0, 1, y.shape[0]),
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
        index = "j"
        velo = "+z"
        pg.compute.pgkokkos.initialize()
        simulate(index, velo)
        pg.compute.pgkokkos.finalize()

    except Exception as e:
        import sys
        import traceback

        print(f"{e}")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        sys.exit(1)
