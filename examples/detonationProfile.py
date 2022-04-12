#!/usr/bin/env python
"""

Generate 1D detonation profile


"""

from mpi4py import MPI
from pathlib import Path

import cantera as ct
import kokkos
import matplotlib.pyplot as plt
import numpy as np
import peregrinepy as pg


def simulate():

    relpath = str(Path(__file__).parent)
    ct.add_directory(relpath + "/../src/peregrinepy/thermo_transport/database/source")
    gas = ct.Solution("CH4_O2_Stanford_Skeletal.yaml")
    # set the gas state
    gas.TP = 300.0, 101325.0
    phi = 1.0
    gas.set_equivalence_ratio(phi, "CH4", "O2")

    config = pg.files.configFile()
    config["RHS"]["diffusion"] = False
    config["RHS"]["shockHandling"] = "hybrid"
    config["RHS"]["primaryAdvFlux"] = "secondOrderKEEP"
    config["RHS"]["secondaryAdvFlux"] = "rusanov"
    config["RHS"]["switchAdvFlux"] = "vanAlbadaPressure"
    config["solver"]["timeIntegration"] = "strang"
    config["thermochem"]["chemistry"] = True
    config["thermochem"]["mechanism"] = "chem_CH4_O2_Stanford_Skeletal"
    config["thermochem"]["eos"] = "tpg"
    config["thermochem"]["spdata"] = "thtr_CH4_O2_Stanford_Skeletal.yaml"
    mb = pg.multiBlock.generateMultiBlockSolver(1, config)

    nx = 500
    dx = 0.005 / 50.0  # Aproximate rde resolution
    lx = nx * dx
    pg.grid.create.multiBlockCube(
        mb,
        mbDims=[1, 1, 1],
        dimsPerBlock=[nx, 2, 2],
        lengths=[lx, 0.01, 0.01],
        periodic=[False, False, False],
    )
    mb.initSolverArrays(config)

    blk = mb[0]
    for face in blk.faces:
        face.bcType = "adiabaticSlipWall"

    mb.setBlockCommunication()
    mb.unifyGrid()
    mb.computeMetrics(config["RHS"]["diffOrder"])

    ng = blk.ng

    q = blk.array["q"]
    q[ng:-ng, ng:-ng, ng:-ng, 0] = gas.P
    q[ng:-ng, ng:-ng, ng:-ng, 4] = gas.T
    q[ng:-ng, ng:-ng, ng:-ng, 5::] = gas.Y[0:-1]

    xc = blk.array["xc"][ng:-ng, ng:-ng, ng:-ng]

    shockX = lx * 0.05
    q[ng:-ng, ng:-ng, ng:-ng, 0] = np.where(
        xc < shockX, 30.0e6, q[ng:-ng, ng:-ng, ng:-ng, 0]
    )
    q[ng:-ng, ng:-ng, ng:-ng, 1] = np.where(
        xc < shockX, 750.0, q[ng:-ng, ng:-ng, ng:-ng, 1]
    )
    q[ng:-ng, ng:-ng, ng:-ng, 4] = np.where(
        xc < shockX, 3000.0, q[ng:-ng, ng:-ng, ng:-ng, 4]
    )
    for i in range(11):
        q[ng:-ng, ng:-ng, ng:-ng, 5 + i] = np.where(
            xc < shockX, 0.0, q[ng:-ng, ng:-ng, ng:-ng, 5 + i]
        )

    # Update cons
    blk.updateDeviceView(["q"])
    mb.eos(blk, mb.thtrdat, 0, "prims")
    # Apply euler boundary conditions
    for face in blk.faces:
        face.bcFunc(blk, face, mb.eos, mb.thtrdat, "euler", mb.tme)
        face.bcFunc(blk, face, mb.eos, mb.thtrdat, "viscous", mb.tme)
    pg.consistify(mb)

    dt = 5.0e-9
    testIndex = int(nx / 2)
    print(mb)
    while blk.array["q"][testIndex, ng, ng, 4] < 350.0:
        if mb.nrt % 10 == 0:
            detLoc = np.where((blk.array["q"][:, ng, ng, 4] > 350.0))[0][-1]
            pg.misc.progressBar(detLoc, testIndex)

        abort = pg.mpiComm.mpiUtils.checkForNan(mb)
        if abort > 0:
            print("Nan")
            break

        mb.step(dt)

    blk.updateHostView(["q"])
    fig, ax1 = plt.subplots()
    ax1.set_title("1D Detonation Profile")
    ax1.set_ylabel("Pressure [MPa]")
    ax1.set_xlabel(r"x")
    x = blk.array["xc"][ng:-ng, ng, ng]
    p = blk.array["q"][ng:-ng, ng, ng, 0] / 1e6
    ax1.plot(x, p, color="r", label="p", linewidth=0.5)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Temperatur[K] / Velocity [m/s]")
    u = blk.array["q"][ng:-ng, ng, ng, 1]
    T = blk.array["q"][ng:-ng, ng, ng, 4]
    ax2.plot(x, T, color="k", label="T", linewidth=0.5)
    ax2.plot(x, u, color="g", label="u", linewidth=0.5)

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
