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
    config["RHS"]["shockHandling"] = None
    config["RHS"]["primaryAdvFlux"] = "rusanov"
    config["solver"]["timeIntegration"] = "chemSubStep"
    config["thermochem"]["chemistry"] = True
    config["thermochem"]["mechanism"] = "chem_CH4_O2_Stanford_Skeletal"
    config["thermochem"]["eos"] = "tpg"
    config["thermochem"]["spdata"] = "thtr_CH4_O2_Stanford_Skeletal.yaml"
    mb = pg.multiBlock.generateMultiBlockSolver(1, config)
    pg.grid.create.multiBlockCube(
        mb,
        mbDims=[1, 1, 1],
        dimsPerBlock=[200, 2, 2],
        lengths=[0.05, 0.01, 0.01],
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
    q[:, :, :, 0] = gas.P
    q[:, :, :, 4] = gas.T
    q[:, :, :, 5::] = gas.Y[0:-1]

    shockX = 0.001
    q[:, :, :, 0] = np.where(blk.array["xc"] < shockX, 60.0e6, q[:, :, :, 0])
    q[:, :, :, 1] = np.where(blk.array["xc"] < shockX, 500, q[:, :, :, 1])
    q[:, :, :, 4] = np.where(blk.array["xc"] < shockX, 5000.0, q[:, :, :, 4])
    for i in range(11):
        q[:, :, :, 5 + i] = np.where(blk.array["xc"] < shockX, 0.0, q[:, :, :, 5 + i])

    pg.writers.writeGrid(mb)
    pg.writers.writeRestart(mb)

    # Update cons
    blk.updateDeviceView(["q"])
    mb.eos(blk, mb.thtrdat, 0, "prims")
    pg.consistify(mb)

    dt = 4.0e-9
    nrtEnd = 3000
    print(mb)
    while mb.nrt < nrtEnd:

        if mb.nrt % 5 == 0:
            pg.misc.progressBar(mb.nrt, nrtEnd)
        abort = pg.mpiComm.mpiUtils.checkForNan(mb)
        if abort > 0:
            print("Nan")
            break

        mb.step(dt)

    blk.updateHostView(["q"])
    fig, ax1 = plt.subplots()
    ax1.set_title("1D Detonation Profile")
    ax1.set_xlabel(r"x")
    x = blk.array["xc"][ng:-ng, ng, ng]
    p = blk.array["q"][ng:-ng, ng, ng, 0] / 1e4
    u = blk.array["q"][ng:-ng, ng, ng, 1]
    T = blk.array["q"][ng:-ng, ng, ng, 4]
    ax1.plot(x, p, color="r", label="p", linewidth=0.5)
    ax1.plot(x, T, color="k", label="T", linewidth=0.5)
    ax1.plot(x, u, color="g", label="u", linewidth=0.5)

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
