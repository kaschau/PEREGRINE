#!/usr/bin/env -S python -m mpi4py
import sys
from mpi4py import MPI
import kokkos
import peregrinepy as pg
import numpy as np
from time import perf_counter

np.seterr(all="raise")


def simulate(configFilePath):

    comm, rank, size = pg.mpiComm.mpiUtils.getCommRankSize()
    if rank == 0:
        string = " >>> ******************************** <<<\n"
        string += "              PEREGRINE CFD\n"
        string += " >>> ******************************** <<<\n"
        print(string)

    config = pg.readers.readConfigFile(configFilePath)
    if rank == 0:
        print("Read config.")

    mb = pg.bootstrapCase(config)

    # Get some stats about the simulation
    nCells = pg.mpiComm.mpiUtils.getNumCells(mb)
    efficiency, slowestProc = pg.mpiComm.mpiUtils.getLoadEfficiency(mb)
    if rank == 0:
        string += " Simulation Summary:\n"
        string += f"  Total cells: {nCells}"
        print(string)
        if efficiency == 100.0:
            print("  Perfect load balancing achieved. 10 points to Gryffindor")
        else:
            print(
                f"  Load Balance Eff: {efficiency: .2f}% (rank {slowestProc})",
            )
        print(mb)
        ts = perf_counter()

    # Time integration
    dt = config["simulation"]["dt"]
    niter = config["simulation"]["niter"]
    niterout = config["simulation"]["niterout"]
    niterprint = config["simulation"]["niterprint"]
    checkNan = config["simulation"]["checkNan"]
    for niter in range(niter):
        dt, CFLmaxA, CFLmaxC = pg.mpiComm.mpiUtils.getDtMaxCFL(mb)
        if mb.nrt % niterprint == 0 and rank == 0:
            print(
                f" >>> --------- nrt: {mb.nrt:<6} ---------- <<<\n",
                f"    tme: {mb.tme:.6E} s\n"
                f"     dt : {dt:.6E} s\n"
                f"     MAX Acoustic   CFL: {CFLmaxA*dt:.3f}\n"
                f"     MAX Convective CFL: {CFLmaxC*dt:.3f}\n"
                " >>> -------------------------------- <<<\n",
            )

        mb.step(dt)

        # Check if we need to output
        if mb.nrt % niterout == 0:
            if rank == 0:
                print("Saving restart.\n")
            pg.writers.parallelWriter.parallelWriteRestart(
                mb, config["io"]["outputdir"]
            )

        # Check if we need to check for Nan
        if checkNan:
            if mb.nrt % checkNan == 0:
                abort = pg.mpiComm.mpiUtils.checkForNan(mb)
                if abort > 0:
                    pg.writers.parallelWriter.parallelWriteRestart(
                        mb, config["io"]["outputdir"]
                    )
                    comm.Barrier()
                    if rank == 0:
                        print("Nan/inf detected. Aborting.")
                        comm.Abort()

        # CoProcess
        mb.coproc(mb)

    # Finalize coprocessor
    mb.coproc.finalize()

    if rank == 0:
        elapsed = perf_counter() - ts
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
