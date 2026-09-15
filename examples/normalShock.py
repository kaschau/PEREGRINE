#!/usr/bin/env python
"""

Generate 1D normal shock


"""

from mpi4py import MPI  # noqa: F401


import matplotlib.pyplot as plt
import numpy as np
import peregrinepy as pg

# a calorically perfect air, stated in full: the library carries no such species
air = {
    "Air": {
        "MW": 28.97,
        "cp0": 1002.838449439523,
        "mu0": 1.8591191080521142e-05,
        "kappa0": 0.02625394405190068,
    }
}


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
    config["timeIntegration"]["integrator"] = "rk4"
    config["mcPhysics"]["eos"] = "cpg"
    config["mcPhysics"]["mixture"] = air
    config.validateConfig()

    nx = 300
    lx = 1.0
    dx = lx / nx
    mb = pg.integrators.getSolver(
        config,
        mesh=pg.mesher.CubeMesher(
            mbDims=[1, 1, 1],
            dimsPerBlock=[nx, 2, 2],
            lengths=[lx, 0.01, 0.01],
            periodic=[False, False, False],
        ),
    )

    blk = mb.blocks[0]
    for face in blk.faces[1::]:
        face.bcType = "adiabaticSlipWall"
    blk.getFace(1).bcType = "constantVelocitySubsonicInlet"

    # We need to get gamma
    ng = blk.ng
    q = blk.q.get()
    q[ng:-ng, ng:-ng, ng:-ng, 0] = p1
    q[ng:-ng, ng:-ng, ng:-ng, 4] = T1
    blk.q.set(q)
    mb.stateFromPrims(nface=0)

    qh = blk.qh.get()
    gamma = qh[ng, ng, ng, 0]
    c1 = qh[ng, ng, ng, 3]
    rho1 = blk.Q.get()[ng, ng, ng, 0]

    # Compute post shock state
    M2 = np.sqrt((M1**2 * (gamma - 1) + 2) / (2 * gamma * M1**2 - (gamma - 1)))
    T2 = T1 * (
        ((1 + (gamma - 1) / 2 * M1**2) * (2 * gamma / (gamma - 1) * M1**2 - 1))
        / (M1**2 * (2 * gamma / (gamma - 1) + (gamma - 1) / 2))
    )
    p2 = p1 * (2 * gamma * M1**2 / (gamma + 1) - (gamma - 1) / (gamma + 1))

    q[ng:-ng, ng:-ng, ng:-ng, 0] = p2
    q[ng:-ng, ng:-ng, ng:-ng, 4] = T2
    blk.q.set(q)
    mb.stateFromPrims(nface=0)
    c2 = blk.qh.get()[ng, ng, ng, 3]

    u2 = -M2 * c2 + M1 * c1  # In lab reference frame
    # Inlet
    valueDict = {"u": u2, "v": 0.0, "w": 0.0, "T": T2}
    face1 = blk.getFace(1)
    face1.bc.setValues(valueDict)

    # Set upstream state
    q[ng:-ng, ng:-ng, ng:-ng, 0] = p1
    q[ng:-ng, ng:-ng, ng:-ng, 4] = T1

    # Set post stock state
    xc = blk.cells.get()[..., 0][ng:-ng, ng:-ng, ng:-ng]

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

    # Update cons
    blk.q.set(q)
    mb.stateFromPrims(nface=0)
    # Apply euler boundary conditions
    mb.applyBcs("euler")
    mb.consistify()

    # Set dt based on cfg estimate
    dt = 0.25 * dx / (c2 + u2)
    testIndex = int(nx / 2)
    print(mb)
    bar = pg.misc.Progress(testIndex)
    while blk.q.get()[testIndex, ng, ng, 4] < 301.0:
        if mb.nrt % 10 == 0:
            shockLoc = np.where((blk.q.get()[:, ng, ng, 4] > 301.0))[0][-1]
            bar.at(shockLoc)

        abort = mb.checkForNan()
        if abort > 0:
            print("Nan")
            break

        mb.step(dt)

    q, Q, qh = blk.q.get(), blk.Q.get(), blk.qh.get()
    fig, ax1 = plt.subplots()
    ax1.set_title("1D Normal Shock")
    ax1.set_ylabel("p/p1")
    ax1.set_xlabel(r"x")
    x = blk.cells.get()[..., 0][ng:-ng, ng, ng]
    p = q[ng:-ng, ng, ng, 0] / p1
    ax1.plot(x, p, color="r", label="p2/p1", linewidth=0.5)
    ax2 = ax1.twinx()
    ax2.set_ylabel("T, M , rho")
    # convert back to shock reference frame
    u = -(q[ng:-ng, ng, ng, 1] - M1 * c1) / qh[ng:-ng, ng, ng, 3]
    T = q[ng:-ng, ng, ng, 4] / T1
    rho = Q[ng:-ng, ng, ng, 0] / rho1
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
