#!/usr/bin/env python
"""

Generate 1D detonation profile


"""

from mpi4py import MPI
from pathlib import Path

import cantera as ct

import matplotlib.pyplot as plt
import numpy as np
import peregrinepy as pg


def simulate():
    relpath = str(Path(__file__).parent)
    ct.add_directory(relpath + "/../src/peregrinepy/thermoTransport/database/source")
    gas = ct.Solution("CH4_O2_Stanford_Skeletal.yaml")
    # set the gas state
    gas.TP = 300.0, 101325.0
    phi = 1.0
    gas.set_equivalence_ratio(phi, "CH4", "O2")

    config = pg.files.configFile()
    config["RHS"]["diffusion"] = False
    config["RHS"]["shockHandling"] = "artificialDissipation"
    config["RHS"]["primaryAdvFlux"] = "KEEPpe"
    config["RHS"]["secondaryAdvFlux"] = "scalarDissipation"
    config["RHS"]["switchAdvFlux"] = "vanLeer"
    config["timeIntegration"]["integrator"] = "rk3"
    config["thermochem"]["chemistry"] = True
    config["thermochem"]["mechanism"] = "chem_CH4_O2_Stanford_Skeletal"
    config["thermochem"]["nChemSubSteps"] = 10
    config["thermochem"]["eos"] = "tpg"
    config["thermochem"]["spdata"] = "thtr_CH4_O2_Stanford_Skeletal.yaml"
    config.validateConfig()
    mb = pg.multiBlock.generateMultiBlockSolver(1, config)

    nx = 300
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
        xc < shockX, 4.0e6, q[ng:-ng, ng:-ng, ng:-ng, 0]
    )
    q[ng:-ng, ng:-ng, ng:-ng, 4] = np.where(
        xc < shockX, 3000.0, q[ng:-ng, ng:-ng, ng:-ng, 4]
    )

    # Update cons
    blk.updateDeviceView(["q"])
    mb.eos(blk, mb.thtrdat, 0, "prims")
    # Apply euler boundary conditions
    for face in blk.faces:
        face.bcFunc(blk, face, mb.eos, mb.thtrdat, "euler", mb.tme)
        face.bcFunc(blk, face, mb.eos, mb.thtrdat, "viscous", mb.tme)
    pg.consistify(mb)

    dt = 1.0e-9
    config["timeIntegration"]["dt"] = dt
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
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.set_title("1D Detonation Profile")
    ax1.set_ylabel("Pressure [MPa]")
    ax1.set_xlabel(r"x")
    x = blk.array["xc"][ng:-ng, ng, ng]
    p = blk.array["q"][ng:-ng, ng, ng, 0] / 1e6
    ax1.plot(x, p, color="r", label="p", linewidth=0.5)
    ax12 = ax1.twinx()
    ax12.set_ylabel("Temperatur[K] / Velocity [m/s]")
    u = blk.array["q"][ng:-ng, ng, ng, 1]
    T = blk.array["q"][ng:-ng, ng, ng, 4]
    ax12.plot(x, T, color="k", label="T", linewidth=0.5)
    ax12.plot(x, u, color="g", label="u", linewidth=0.5)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax12.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2)

    O2 = blk.array["q"][ng:-ng, ng, ng, 5 + 2]
    H2O = blk.array["q"][ng:-ng, ng, ng, 5 + 6]
    CO2 = 1.0 - np.sum(blk.array["q"][ng:-ng, ng, ng, 5::], axis=1)
    CH4 = blk.array["q"][ng:-ng, ng, ng, 5 + 8]
    ax2.plot(x, O2, color="b", label="O2", linewidth=0.5)
    ax2.plot(x, CH4, color="r", label="CH4", linewidth=0.5)
    ax2.plot(x, H2O, color="k", label="H2O", linewidth=0.5)
    ax2.plot(x, CO2, color="g", label="CO2", linewidth=0.5)
    ax2.legend()
    plt.show()
    plt.close()


if __name__ == "__main__":
    try:
        pg.compute.pgkokkos.initialize()
        simulate()
        pg.compute.pgkokkos.finalize()

    except Exception as e:
        import sys
        import traceback

        print(f"{e}")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        sys.exit(1)
