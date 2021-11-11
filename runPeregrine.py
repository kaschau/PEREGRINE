#!/usr/bin/env -S python -m mpi4py
import sys
import kokkos
from mpi4py import MPI
import peregrinepy as pg
import numpy as np
import time

np.seterr(all="raise")


def simulate(configFilePath):

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

    dt = config["simulation"]["dt"]
    niter = config["simulation"]["niter"]
    niterout = config["simulation"]["niterout"]
    for niter in range(niter):

        if mb.nrt % config["simulation"]["niterprint"] == 0:
            cfl = np.array(pg.compute.utils.CFLmax(mb))
            comm.Allreduce(MPI.IN_PLACE, cfl, op=MPI.MAX)
            if rank == 0:
                print(
                    " >>> -------------------------------- <<<\n",
                    f"    nrt: {mb.nrt:6>}, tme: {mb.tme:.6E}\n"
                    f"     dt:{dt:.6E}\n"
                    f"     MAX Acoustic   CFL: {cfl[0]*dt:.3E}\n"
                    f"     MAX Convective CFL: {cfl[1]*dt:.3E}\n"
                    " >>> -------------------------------- <<<\n",
                )

        mb.step(dt)

        if mb.nrt % niterout == 0:
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
            f"Seconds/Iteration/Cell: {elapsed/config['simulation']['niter']/nCells}\n"
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
