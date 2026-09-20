#!/usr/bin/env python
"""

Generate 1D detonation profile


"""

from mpi4py import MPI  # noqa: F401
from pathlib import Path

import cantera as ct

import matplotlib.pyplot as plt
import numpy as np
import peregrinepy as pg


def simulate():
    relpath = str(Path(__file__).parent)
    ct.add_directory(relpath + "/../src/peregrinepy/mixture/database/mechanisms")
    gas = ct.Solution("CH4_O2_FFCMY.yaml")
    # set the gas state
    gas.TP = 300.0, 101325.0
    phi = 1.0
    gas.set_equivalence_ratio(phi, "CH4", "O2")

    config = pg.files.configFile()
    config["simulation"]["physics"] = "euler"
    config["RHS"]["primaryAdvFlux"] = "KEPaEC"
    config["RHS"]["secondaryAdvFlux"] = "rusanov"
    config["RHS"]["switchAdvFlux"] = "jamesonPressure"
    config["RHS"]["switchValues"] = {"gain": 5.0}
    config["timeIntegration"]["integrator"] = "rk3"
    config["simulation"]["chemistry"] = "substepped"
    config["simulation"]["eos"] = "tpg"
    config["simulation"]["mixture"] = "CH4_O2_FFCMY.yaml"
    config["simulation"]["Trange"] = (300.0, 3500.0)
    config["bcValues"]["walls"] = {"bcType": "adiabaticSlipWall"}
    config.validateConfig()

    nx = 300
    dx = 0.005 / 50.0  # Aproximate rde resolution
    lx = nx * dx
    mesh = pg.mesher.CubeMesher(
        mbDims=[1, 1, 1],
        dimsPerBlock=[nx, 2, 2],
        lengths=[lx, 0.01, 0.01],
        boundaryNames=dict.fromkeys(range(1, 7), "walls"),
    )
    mb = pg.multiBlock.solver(config, mesh)

    blk = mb.blocks[0]
    ng = blk.ng

    # the primitive vector, p, u, v, w, T, Y: the mixture at rest, and hot
    # and pressed behind the shock
    q = np.zeros(blk.Q.shape[:3] + (mb.ne,))
    q[:, :, :, 0] = gas.P
    q[:, :, :, 4] = gas.T
    q[:, :, :, 5::] = gas.Y[0:-1]
    xc = blk.cells.get()[..., 0]
    shockX = lx * 0.05
    q[:, :, :, 0] = np.where(xc < shockX, 4.0e6, q[:, :, :, 0])
    q[:, :, :, 4] = np.where(xc < shockX, 3000.0, q[:, :, :, 4])
    mb.setPrimitives([q])

    dt = 1.0e-9
    config["timeIntegration"]["dt"] = dt
    testIndex = int(nx / 2)
    print(mb)
    bar = pg.misc.Progress(testIndex)
    T = lambda: mb.exportData(blk, ["T"])["T"][:, ng, ng]
    while T()[testIndex] < 350.0:
        if mb.nrt % 10 == 0:
            detLoc = np.where(T() > 350.0)[0][-1]
            bar.at(detLoc)
        if not np.isfinite(blk.Q.get()[blk.interior]).all():
            print("Nan")
            break

        mb.integrator.step(dt)

    data = mb.exportData(blk, ["p", "u", "T", "O2", "H2O", "CO2", "CH4"])
    line = lambda name: data[name][ng:-ng, ng, ng]
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.set_title("1D Detonation Profile")
    ax1.set_ylabel("Pressure [MPa]")
    ax1.set_xlabel(r"x")
    x = blk.cells.get()[..., 0][ng:-ng, ng, ng]
    p = line("p") / 1e6
    ax1.plot(x, p, color="r", label="p", linewidth=0.5)
    ax12 = ax1.twinx()
    ax12.set_ylabel("Temperatur[K] / Velocity [m/s]")
    u = line("u")
    T = line("T")
    ax12.plot(x, T, color="k", label="T", linewidth=0.5)
    ax12.plot(x, u, color="g", label="u", linewidth=0.5)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax12.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2)

    O2, H2O, CO2, CH4 = (line(name) for name in ("O2", "H2O", "CO2", "CH4"))
    ax2.plot(x, O2, color="b", label="O2", linewidth=0.5)
    ax2.plot(x, CH4, color="r", label="CH4", linewidth=0.5)
    ax2.plot(x, H2O, color="k", label="H2O", linewidth=0.5)
    ax2.plot(x, CO2, color="g", label="CO2", linewidth=0.5)
    ax2.legend()
    plt.show()
    plt.close()


if __name__ == "__main__":
    try:
        pg.backend.abi.lib.initialize()
        simulate()
        pg.backend.abi.lib.finalize()

    except Exception as e:
        import sys
        import traceback

        print(f"{e}")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        sys.exit(1)
