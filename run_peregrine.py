#!/usr/bin/env -S python -m mpi4py
import sys
import kokkos
import peregrinepy as pg
import numpy as np

np.seterr(all="raise")


def simulate(configFilePath):
    # Import but do not initialise MPI

    comm, rank, size = pg.mpicomm.mpiutils.getCommRankSize()

    config = pg.mpicomm.mpiReadConfig(configFilePath)

    mb = pg.bootstrapCase(config)

    for niter in range(config["simulation"]["niter"]):

        if mb.nrt % config["simulation"]["niterprint"] == 0:
            if rank == 0:
                print(
                    " >>> -------------------------------- <<<\n",
                    f"nrt: {mb.nrt:6>}, tme: {mb.tme:.6E}\n"
                    " >>> -------------------------------- <<<\n",
                )

        if mb.nrt % config["simulation"]["niterout"] == 0:
            if rank == 0:
                print("Saving restart.\n")
            pg.writers.parallelWriter.parallelWriteRestart(
                mb, config["io"]["outputdir"]
            )

        mb.step(config["simulation"]["dt"])


if __name__ == "__main__":
    configFilePath = sys.argv[1]
    try:
        # Manually initialise MPI
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
