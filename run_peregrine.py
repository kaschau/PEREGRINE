#!/usr/bin/env python
import mpi4py.rc

mpi4py.rc.initialize = False

import kokkos

import peregrinepy as pg
import time

import sys


def simulate(configFilePath):
    # Import but do not initialise MPI
    from mpi4py import MPI
    import numpy as np

    np.seterr(all="raise")

    # Manually initialise MPI
    MPI.Init()

    comm, rank, size = pg.mpicomm.mpiutils.getCommRankSize()
    # Ensure MPI is suitably cleaned up
    pg.mpicomm.mpiutils.registerFinalizeHandler()

    config = pg.mpicomm.mpiReadConfig(configFilePath)

    mb = pg.bootstrapCase(config)

    pg.writers.parallelWriter.parallelWriteRestart(
        mb, path=config["io"]["outputdir"]
    )

    for niter in range(config["simulation"]["niter"]):

        if mb.nrt % config["simulation"]["niterprint"] == 0:
            if rank == 0:
                print(
                    " >>> -------------------------------- <<<\n",
                    f"nrt: {mb.nrt:6>}, tme: {mb.tme:.6E}\n"
                    " >>> -------------------------------- <<<\n",
                )

        mb.step(config["simulation"]["dt"])

        if mb.nrt % config["simulation"]["niterout"] == 0:
            if rank == 0:
                print("Saving restart.\n")
            pg.writers.parallelWriter.parallelWriteRestart(
                mb, config["io"]["outputdir"]
            )

    # Finalise MPI
    MPI.Finalize()


if __name__ == "__main__":
    configFilePath = sys.argv[1]
    try:
        kokkos.initialize()
        simulate(configFilePath)
        kokkos.finalize()

    except Exception as e:
        import sys
        import traceback

        print(f"{e}")
        excType, excValue, excTraceback = sys.exc_info()
        traceback.print_exception(excType, excValue, excTraceback)
        sys.exit(1)
