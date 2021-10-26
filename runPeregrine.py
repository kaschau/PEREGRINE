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

    # Get some stats about the simulation
    nCells = pg.mpiComm.mpiUtils.getNumCells(mb)
    slowest, slowestProc = pg.mpiComm.mpiUtils.getLoadEfficiency(mb)
    if rank == 0:
        string = " >>> ******************************** <<<\n"
        string += "              PEREGRINE CFD\n"
        string += " >>> ******************************** <<<\n"
        string += " Simulation Summary:"
        print(string)
        if slowest == 100.0:
            print("  Perfect load balancing achieved. 10 points to Gryffindor")
        else:
            print(
                f"  Load Balance Eff: {slowest: .2f}% (rank {slowestProc})",
            )
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
        print(
            "PEREGRINE simulation completed.\n"
            f"Simulation time: {hrs}h : {mins}m : {int(secs)}s\n"
            f"Time/Iteration/Cell: {elapsed/config['simulation']['niter']/nCells}\n"
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
