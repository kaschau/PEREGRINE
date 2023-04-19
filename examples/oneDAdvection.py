#!/usr/bin/env python
"""

Test case from

Preventing spurious pressure oscillations in split convective form discretization for compressible flows
https://doi.org/10.1016/j.jcp.2020.110060

Should reproduce results in Fig. 1 for the KEEP scheme (blue line)


"""

from mpi4py import MPI

import peregrinepy as pg
import numpy as np
import matplotlib.pyplot as plt


def simulate(index="i"):
    config = pg.files.configFile()
    config["RHS"]["diffusion"] = False
    config.validateConfig()
    mb = pg.multiBlock.generateMultiBlockSolver(1, config)
    print(mb)

    rot = {"i": 0, "j": 1, "k": 2}

    def rotate(li, index):
        return li[-rot[index] :] + li[: -rot[index]]

    nx = 41
    dimsPerBlock = rotate([nx, 2, 2], index)
    lengths = rotate([1, 0.1, 0.1], index)
    periodic = rotate([True, False, False], index)

    pg.grid.create.multiBlockCube(
        mb,
        mbDims=[1, 1, 1],
        dimsPerBlock=dimsPerBlock,
        lengths=lengths,
        periodic=periodic,
    )
    mb.initSolverArrays(config)

    blk = mb[0]
    for face in blk.faces:
        if face.bcType.endswith("Wall"):
            face.bcType = "adiabaticSlipWall"
    if index == "i":
        blk.getFace(1).commRank = 0
        blk.getFace(2).commRank = 0
    elif index == "j":
        blk.getFace(3).commRank = 0
        blk.getFace(4).commRank = 0
    elif index == "k":
        blk.getFace(5).commRank = 0
        blk.getFace(6).commRank = 0

    mb.setBlockCommunication()
    mb.unifyGrid()
    mb.computeMetrics(config["RHS"]["diffOrder"])

    ng = blk.ng
    R = 287.002507
    cp = 1000.0
    gamma = cp / (cp - R)
    ccArray = {"i": "xc", "j": "yc", "k": "zc"}
    uIndex = {"i": 1, "j": 2, "k": 3}
    blk.array["q"][:, :, :, 0] = 1.0
    blk.array["q"][:, :, :, uIndex[index]] = 1.0
    xc = blk.array[ccArray[index]]
    initial_rho = 2.0 + np.sin(2 * np.pi * xc)
    initial_T = 1.0 / (R * initial_rho)
    blk.array["q"][:, :, :, 4] = initial_T

    # Update cons
    blk.updateDeviceView(["q", "Q"])
    mb.eos(blk, mb.thtrdat, 0, "prims")
    pg.consistify(mb)

    # entropy stuff
    blk.array["s"][:] = blk.array["Q"][:, :, :, 0] * np.log(
        blk.array["q"][:, :, :, 0] * blk.array["Q"][:, :, :, 0] ** (-gamma)
    )
    blk.updateDeviceView(["s"])
    pg.consistify(mb)

    sDerived = []
    sEvolved = []
    t = []
    dt = 0.1 * 0.0025
    tEnd = 11.0
    while mb.tme < tEnd:
        if mb.nrt % 50 == 0:
            pg.misc.progressBar(mb.tme, tEnd)

        if mb.nrt % 10 == 0:
            dS = np.sum(
                blk.array["Q"][ng:-ng, ng:-ng, ng:-ng, 0]
                * np.log(
                    blk.array["q"][ng:-ng, ng:-ng, ng:-ng, 0]
                    * blk.array["Q"][ng:-ng, ng:-ng, ng:-ng, 0] ** (-gamma)
                )
            )
            sDerived.append(dS)
            eS = np.sum(blk.array["s"][ng:-ng, ng:-ng, ng:-ng])
            sEvolved.append(eS)
            t.append(mb.tme)

        mb.step(dt)

    blk.updateHostView(["q", "Q"])
    fig, ax1 = plt.subplots()
    ax1.set_title("1D Advection Results")
    ax1.set_xlabel(r"x")
    s_ = rotate(np.s_[ng:-ng, ng, ng], index)
    x = blk.array[ccArray[index]][s_]
    rho = blk.array["Q"][s_][:, 0]
    p = blk.array["q"][s_][:, 0]
    u = blk.array["q"][s_][:, uIndex[index]]
    sd = rho * np.log(p * rho**-gamma)
    se = blk.array["s"][s_]
    ax1.plot(x, rho, color="g", label="rho", linewidth=0.5)
    ax1.plot(x, p, color="r", label="p", linewidth=0.5)
    ax1.plot(x, u, color="k", label="u", linewidth=0.5)
    ax1.scatter(
        x,
        initial_rho[ng:-ng, ng:-ng, ng:-ng],
        marker="o",
        facecolor="w",
        edgecolor="b",
        label="exact",
        linewidth=0.5,
    )
    ax1.legend()
    plt.show()
    plt.clf()
    # entropy
    plt.plot(x, sd, color="k", label="sd", linewidth=0.5)
    plt.plot(x, se, color="r", label="se", linewidth=0.5)
    plt.legend()
    plt.show()
    plt.clf()
    plt.plot(t, (-(sDerived - sDerived[0])) / sDerived[0], label="Derived")
    plt.scatter(
        t, (-(sEvolved - sEvolved[0])) / sEvolved[0], marker="o", label="Evolved"
    )
    plt.legend()
    plt.title(r"$\Delta(\rho s) / (\rho_0 s_0)$")
    plt.show()
    plt.close()


if __name__ == "__main__":
    try:
        # a = pg.compute.pgkokkos.ScopeGuard()
        pg.compute.pgkokkos.initialize()
        simulate("i")
        pg.compute.pgkokkos.finalize()

    except Exception as e:
        import sys
        import traceback

        print(f"{e}")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        sys.exit(1)
