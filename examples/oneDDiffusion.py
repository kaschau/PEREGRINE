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
    config["mcPhysics"]["mixture"] = ["O2", "N2"]
    config["mcPhysics"]["eos"] = "tpg"
    config["mcPhysics"]["trans"] = "kineticTheory"
    config["mcPhysics"]["Trange"] = (200.0, 1000.0)
    config["RHS"]["diffusion"] = True
    config["RHS"]["primaryAdvFlux"] = "rusanov"
    config.validateConfig()
    mb = pg.integrators.getSolver(
        config,
        mesh=pg.mesher.CubeMesher(
            mbDims=[1, 1, 1], dimsPerBlock=[41, 2, 2], lengths=[1, 0.01, 0.01]
        ),
    )

    blk = mb.blocks[0]
    for face in blk.faces:
        face.bcType = "adiabaticSlipWall"

    ng = blk.ng
    q = blk.q.get()
    q[:, :, :, 0] = 101325.0
    # Make equal mass
    MWA, MWB = mb.thtrdat.MW.get()[:2]
    xc = blk.cells.get()[..., 0]
    q[:, :, :, 4] = np.where(xc < 0.5, 300.0 * MWA / MWB, 300.0)
    q[:, :, :, 5] = np.where(xc < 0.5, 1.0, 0.0)

    # Update cons
    blk.q.set(q)
    mb.stateFromPrims(nface=0)
    mb.consistify()

    dt = 1e-5
    nrt = 50000
    bar = pg.misc.Progress(nrt)
    while mb.nrt < nrt:
        mb.step(dt)
        if mb.nrt % 100 == 0:
            bar.at(mb.nrt)

    q = blk.q.get()
    fig, ax1 = plt.subplots()
    ax1.set_title("1D Diffusion Results")
    ax1.set_xlabel(r"x")
    x = blk.cells.get()[..., 0][ng:-ng, ng, ng]
    A = q[ng:-ng, ng, ng, 5]
    B = 1.0 - q[ng:-ng, ng, ng, 5]
    ax1.plot(x, A, marker="o", color="r", label="A", linewidth=1.0)
    ax1.plot(x, B, linestyle="--", color="k", label="B", linewidth=1.5)
    ax1.set_ylim([0.49, 0.51])
    ax1.legend()
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
