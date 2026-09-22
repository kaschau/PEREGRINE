#!/usr/bin/env -S python -m mpi4py
"""

Binary diffusion between two species of equal total mass inside box

"""

from mpi4py import MPI  # noqa: F401

import peregrinepy as pg
import numpy as np
import matplotlib.pyplot as plt


def simulate():
    config = pg.files.configFile()
    config["simulation"]["simulator"] = "navierStokes"
    config["mixture"]["species"] = ["O2", "N2"]
    config["mixture"]["eos"] = "tpg"
    config["mixture"]["trans"] = "kineticTheory"
    config["mixture"]["Trange"] = (200.0, 1000.0)
    config["RHS"]["primaryAdvFlux"] = "rusanov"
    config["bcValues"]["walls"] = {"bcType": "adiabaticSlipWall"}
    config.validateConfig()
    mesh = pg.mesher.CubeMesher(
        mbDims=[1, 1, 1],
        dimsPerBlock=[41, 2, 2],
        lengths=[1, 0.01, 0.01],
        boundaryNames=dict.fromkeys(range(1, 7), "walls"),
    )
    mb = pg.multiBlock.solver(config, mesh)

    blk = mb.blocks[0]
    ng = blk.ng
    # the primitive vector, p, u, v, w, T, Y(O2): one species each side of
    # the middle, the temperatures making the masses equal
    q = np.zeros(blk.Q.shape[:3] + (mb.ne,))
    q[:, :, :, 0] = 101325.0
    MWA, MWB = mb.simulator.mixture.speciesData()["MW"][:2]
    xc = blk.cells.get()[..., 0]
    q[:, :, :, 4] = np.where(xc < 0.5, 300.0 * MWA / MWB, 300.0)
    q[:, :, :, 5] = np.where(xc < 0.5, 1.0, 0.0)
    mb.setPrimitives([q])

    dt = 1e-5
    nrt = 50000
    bar = pg.misc.Progress(nrt)
    while mb.nrt < nrt:
        mb.integrator.step(dt)
        if mb.nrt % 100 == 0:
            bar.at(mb.nrt)

    A = mb.exportData(blk, ["O2"])["O2"]
    fig, ax1 = plt.subplots()
    ax1.set_title("1D Diffusion Results")
    ax1.set_xlabel(r"x")
    x = blk.cells.get()[..., 0][ng:-ng, ng, ng]
    A = A[ng:-ng, ng, ng]
    B = 1.0 - A
    ax1.plot(x, A, marker="o", color="r", label="A", linewidth=1.0)
    ax1.plot(x, B, linestyle="--", color="k", label="B", linewidth=1.5)
    ax1.set_ylim([0.49, 0.51])
    ax1.legend()
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
