#!/usr/bin/env python
"""

Couette Flow with top wall moving at 5m/s

"""

from mpi4py import MPI  # noqa: F401

import peregrinepy as pg
import numpy as np
import matplotlib.pyplot as plt

# a calorically perfect air, stated in full: the library carries no such species
air = {
    "Air": {
        "MW": 28.97,
        "cp0": 1002.838449439523,
        "mu0": 1.8591191080521142e-05,
        "kappa0": 0.02625394405190068,
    }
}


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
    config["mcPhysics"]["trans"] = "constantProps"
    config["mcPhysics"]["mixture"] = air
    config.validateConfig()

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

    mb = pg.integrators.getSolver(
        config,
        mesh=pg.mesher.CubeMesher(
            mbDims=[1, 1, 1],
            dimsPerBlock=dimsPerBlock,
            lengths=lengths,
            periodic=periodic,
        ),
    )

    blk = mb.blocks[0]

    if index == "i":
        blk.getFace(1).bcType = "adiabaticNoSlipWall"
        blk.getFace(2).bcType = "adiabaticMovingWall"
        if "y" in velo:
            for face in [5, 6]:
                blk.getFace(face).bcType = "adiabaticSlipWall"
        else:
            for face in [3, 4]:
                blk.getFace(face).bcType = "adiabaticSlipWall"
    elif index == "j":
        blk.getFace(3).bcType = "adiabaticNoSlipWall"
        blk.getFace(4).bcType = "adiabaticMovingWall"
        if "x" in velo:
            for face in [5, 6]:
                blk.getFace(face).bcType = "adiabaticSlipWall"
        else:
            for face in [1, 2]:
                blk.getFace(face).bcType = "adiabaticSlipWall"
    elif index == "k":
        blk.getFace(5).bcType = "adiabaticNoSlipWall"
        blk.getFace(6).bcType = "adiabaticMovingWall"
        if "x" in velo:
            for face in [3, 4]:
                blk.getFace(face).bcType = "adiabaticSlipWall"
        else:
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
            face.bc.setValues(valueDict)
            break

    ng = blk.ng
    # the wall's values changed since the case was made, so its halos follow
    mb.consistify()

    mu = np.unique(mb.thtrdat.mu0.get())[0]
    rho = np.unique(blk.Q.get()[:, :, :, 0])[0]
    nu = mu / rho

    if index == "i":
        s_ = np.s_[ng:-ng, ng, ng]
    elif index == "j":
        s_ = np.s_[ng, ng:-ng, ng]
    elif index == "k":
        s_ = np.s_[ng, ng, ng:-ng]
    ccAxis = {"i": 0, "j": 1, "k": 2}
    if "x" in velo:
        uIndex = 1
    elif "y" in velo:
        uIndex = 2
    elif "z" in velo:
        uIndex = 3
    else:
        raise ValueError()

    xc = blk.cells.get()[..., ccAxis[index]][s_]
    sU_ = s_ + (uIndex,)

    outputTimes = [0.0005, 0.005, 0.05]
    doneOutput = [False for _ in range(len(outputTimes))]
    outputU = []
    simTme = max(outputTimes) * h**2 / nu
    bar = pg.misc.Progress(simTme)
    while mb.tme < simTme:
        mb.step(config["timeIntegration"]["dt"])

        if mb.nrt % 200 == 0:
            bar.at(mb.tme)
            if np.any(np.isnan(blk.Q.get())):
                raise ValueError("Nan detected")

        for i, oT in enumerate(outputTimes):
            t = oT * h**2 / nu
            if mb.tme >= t and not doneOutput[i]:
                outputU.append(blk.q.get()[sU_])
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
        pg.abi.lib.initialize()
        simulate(index, velo)
        pg.abi.lib.finalize()

    except Exception as e:
        import sys
        import traceback

        print(f"{e}")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        sys.exit(1)
