#!/usr/bin/env python
"""

Generate 1D normal shock


"""

from mpi4py import MPI

import kokkos
import matplotlib.pyplot as plt
import numpy as np
import peregrinepy as pg


##################################################
######### 1D Normal Shock ########################
##################################################

# Set upstream values here
M1 = 2.0
p1 = 101325.0
T1 = 300.0


def simulate():

    config = pg.files.configFile()
    config["RHS"]["diffusion"] = False

    config["RHS"]["primaryAdvFlux"] = "rusanov"

    config["solver"]["timeIntegration"] = "rk4"
    config["thermochem"]["eos"] = "cpg"
    config["thermochem"]["spdata"] = ["Air"]
    mb = pg.multiBlock.generateMultiBlockSolver(1, config)

    nx = 300
    lx = 1.0
    dx = lx / nx
    pg.grid.create.multiBlockCube(
        mb,
        mbDims=[1, 1, 1],
        dimsPerBlock=[nx, 2, 2],
        lengths=[lx, 0.01, 0.01],
        periodic=[False, False, False],
    )
    mb.initSolverArrays(config)

    blk = mb[0]
    for face in blk.faces[1::]:
        face.bcType = "adiabaticSlipWall"
    blk.getFace(1).bcType = "constantVelocitySubsonicInlet"

    # We need to get gamma
    ng = blk.ng
    q = blk.array["q"]
    q[ng:-ng, ng:-ng, ng:-ng, 0] = p1
    q[ng:-ng, ng:-ng, ng:-ng, 4] = T1
    mb.eos(blk, mb.thtrdat, 0, "prims")

    gamma = blk.array["qh"][ng, ng, ng, 0]
    c1 = blk.array["qh"][ng, ng, ng, 3]
    rho1 = blk.array["Q"][ng, ng, ng, 0]

    # Compute post shock state
    M2 = np.sqrt((M1 ** 2 * (gamma - 1) + 2) / (2 * gamma * M1 ** 2 - (gamma - 1)))
    T2 = T1 * (
        ((1 + (gamma - 1) / 2 * M1 ** 2) * (2 * gamma / (gamma - 1) * M1 ** 2 - 1))
        / (M1 ** 2 * (2 * gamma / (gamma - 1) + (gamma - 1) / 2))
    )
    p2 = p1 * (2 * gamma * M1 ** 2 / (gamma + 1) - (gamma - 1) / (gamma + 1))

    q[ng:-ng, ng:-ng, ng:-ng, 0] = p2
    q[ng:-ng, ng:-ng, ng:-ng, 4] = T2
    mb.eos(blk, mb.thtrdat, 0, "prims")
    c2 = blk.array["qh"][ng, ng, ng, 3]

    u2 = -M2 * c2 + M1 * c1  # In lab reference frame
    # Inlet
    valueDict = {"u": u2, "v": 0.0, "w": 0.0, "T": T2}
    face1 = blk.getFace(1)
    face1.array["qBcVals"] = np.zeros(blk.array["q"][face1.s1_].shape)
    pg.bcs.prepInlets.prep_constantVelocitySubsonicInlet(blk, face1, valueDict)
    pg.misc.createViewMirrorArray(face1, "qBcVals", blk.array["q"][face1.s1_].shape)

    mb.setBlockCommunication()
    mb.unifyGrid()
    mb.computeMetrics(config["RHS"]["diffOrder"])

    # Set upstream state
    q[ng:-ng, ng:-ng, ng:-ng, 0] = p1
    q[ng:-ng, ng:-ng, ng:-ng, 4] = T1

    # Set post stock state
    xc = blk.array["xc"][ng:-ng, ng:-ng, ng:-ng]

    shockX = lx * 0.05
    q[ng:-ng, ng:-ng, ng:-ng, 0] = np.where(
        xc < shockX, p2, q[ng:-ng, ng:-ng, ng:-ng, 0]
    )
    q[ng:-ng, ng:-ng, ng:-ng, 1] = np.where(
        xc < shockX, u2, q[ng:-ng, ng:-ng, ng:-ng, 1]
    )
    q[ng:-ng, ng:-ng, ng:-ng, 4] = np.where(
        xc < shockX, T2, q[ng:-ng, ng:-ng, ng:-ng, 4]
    )

    # # Update cons
    mb.eos(blk, mb.thtrdat, 0, "prims")
    # Apply euler boundary conditions
    for face in blk.faces:
        face.bcFunc(blk, face, mb.eos, mb.thtrdat, "euler", mb.tme)
        face.bcFunc(blk, face, mb.eos, mb.thtrdat, "viscous", mb.tme)
    pg.consistify(mb)

    # Set dt based on cfg estimate
    dt = 0.25 * dx / (c2 + u2)
    testIndex = int(nx / 2)
    print(mb)
    while blk.array["q"][testIndex, ng, ng, 4] < 301.0:
        if mb.nrt % 10 == 0:
            shockLoc = np.where((blk.array["q"][:, ng, ng, 4] > 301.0))[0][-1]
            pg.misc.progressBar(shockLoc, testIndex)

        abort = pg.mpiComm.mpiUtils.checkForNan(mb)
        if abort > 0:
            print("Nan")
            break

        mb.step(dt)

    fig, ax1 = plt.subplots()
    ax1.set_title("1D Normal Shock")
    ax1.set_ylabel("p/p1")
    ax1.set_xlabel(r"x")
    x = blk.array["xc"][ng:-ng, ng, ng]
    p = blk.array["q"][ng:-ng, ng, ng, 0] / p1
    ax1.plot(x, p, color="r", label="p2/p1", linewidth=0.5)
    ax2 = ax1.twinx()
    ax2.set_ylabel("T, M , rho")
    # convert back to shock reference frame
    u = (
        -(blk.array["q"][ng:-ng, ng, ng, 1] - M1 * c1)
        / blk.array["qh"][ng:-ng, ng, ng, 3]
    )
    T = blk.array["q"][ng:-ng, ng, ng, 4] / T1
    rho = blk.array["Q"][ng:-ng, ng, ng, 0] / rho1
    ax2.plot(x, T, color="k", label="T/T1", linewidth=0.5)
    ax2.plot(x, u, color="g", label="M", linewidth=0.5)
    ax2.plot(x, rho, color="orange", label="rho/rho1", linewidth=0.5)
    ax2.set_ylim([0, None])

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2)
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
