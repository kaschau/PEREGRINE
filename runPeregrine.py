#!/usr/bin/env -S python -m mpi4py
import sys
import kokkos
import peregrinepy as pg
import numpy as np
import time

np.seterr(all="raise")


def simulate(configFilePath):
    # Import but do not initialise MPI

    comm, rank, size = pg.mpiComm.mpiUtils.getCommRankSize()

    config = pg.mpiComm.mpiReadConfig(configFilePath)

    mb = pg.bootstrapCase(config)

    if rank == 0:
        print(mb)
        ts = time.time()

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

    if rank == 0:
        elapsed = time.time() - ts
        hrs, rem = divmod(elapsed, 3600.0)
        mins, secs = divmod(rem, 60.0)
        print("PEREGRINE simulation completed.\n"
              f"Simulation time: {hrs}H:{mins}M:{secs}S\n "
              )


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
