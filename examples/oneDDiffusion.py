#!/usr/bin/env -S python -m mpi4py
"""

Test case from

Preventing spurious pressure oscillations in split convective form discretization for compressible flows
https://doi.org/10.1016/j.jcp.2020.110060

Should reproduce results in Fig. 1 for the KEEP scheme (blue line)


"""

from mpi4py import MPI
import kokkos
import peregrinepy as pg
import numpy as np
import matplotlib.pyplot as plt


def simulate():

    config = pg.files.configFile()
    config["thermochem"]["spdata"] = "ab-cpg.yaml"
    config["RHS"]["diffusion"] = True
    config["RHS"]["primaryAdvFlux"] = "rusanov"
    mb = pg.multiBlock.generateMultiBlockSolver(1, config)
    pg.grid.create.multiBlockCube(
        mb,
        mbDims=[1, 1, 1],
        dimsPerBlock=[41, 2, 2],
        lengths=[1, 0.01, 0.01],
    )
    mb.initSolverArrays(config)

    blk = mb[0]
    for face in blk.faces:
        face.bcType = "adiabaticSlipWall"

    mb.setBlockCommunication()

    mb.unifyGrid()
    mb.computeMetrics(config["RHS"]["diffOrder"])

    ng = blk.ng
    blk.array["q"][:, :, :, 0] = 101325.0
    # Make equal mass
    MWA = mb.thtrdat.array["MW"][0]
    MWB = mb.thtrdat.array["MW"][1]
    blk.array["q"][:, :, :, 4] = np.where(
        blk.array["xc"] < 0.5, 300.0 * MWA / MWB, 300.0
    )
    blk.array["q"][:, :, :, 5] = np.where(blk.array["xc"] < 0.5, 1.0, 0.0)

    # Update cons
    mb.eos(blk, mb.thtrdat, 0, "prims")
    pg.consistify(mb)

    dt = 1e-5
    nrt = 50000
    while mb.nrt < nrt:
        mb.step(dt)
        if mb.nrt % 100 == 0:
            pg.misc.progressBar(mb.nrt, nrt)

    fig, ax1 = plt.subplots()
    ax1.set_title("1D Diffusion Results")
    ax1.set_xlabel(r"x")
    x = blk.array["xc"][ng:-ng, ng, ng]
    A = blk.array["q"][ng:-ng, ng, ng, 5]
    B = 1.0 - blk.array["q"][ng:-ng, ng, ng, 5]
    ax1.plot(x, A, marker="o", color="r", label="A", linewidth=1.0)
    ax1.plot(x, B, linestyle="--", color="k", label="B", linewidth=1.5)
    ax1.set_ylim([0.49, 0.51])
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
