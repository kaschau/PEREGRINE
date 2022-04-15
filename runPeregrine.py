#!/usr/bin/env -S python -m mpi4py
import sys
from mpi4py import MPI
from time import perf_counter

import kokkos
import numpy as np

import peregrinepy as pg

np.seterr(all="raise")


def simulate(configFilePath):

    comm, rank, size = pg.mpiComm.mpiUtils.getCommRankSize()
    if rank == 0:
        string = " >>> ******************************** <<<\n"
        string += "              PEREGRINE CFD\n"
        string += " >>> ******************************** <<<\n"
        string += "  Copyright (c) 2021-2022 Kyle A. Schau\n"
        string += "           All rights reserved.\n"
        print(string)

    config = pg.readers.readConfigFile(configFilePath, parallel=True)
    if rank == 0:
        print("Read config.")

    mb = pg.bootstrapCase(config)

    # Get some stats about the simulation
    nCells = pg.mpiComm.mpiUtils.getNumCells(mb)
    efficiency, slowestProc = pg.mpiComm.mpiUtils.getLoadEfficiency(mb)
    if rank == 0:
        string = " Simulation Summary:\n"
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
    niter = config["simulation"]["niter"]
    niterRestart = config["simulation"]["niterRestart"]
    niterArchive = config["simulation"]["niterArchive"]
    niterPrint = config["simulation"]["niterPrint"]
    checkNan = config["simulation"]["checkNan"]
    for niter in range(niter):
        dt, CFLmaxA, CFLmaxC, CFLmax = pg.mpiComm.mpiUtils.getDtMaxCFL(mb)
        if mb.nrt % niterPrint == 0 and rank == 0:
            print(
                f" >>> --------- nrt: {mb.nrt:<6} ---------- <<<\n",
                f"    tme: {mb.tme:.6E} s\n"
                f"     dt : {dt:.6E} s\n"
                f"     MAX CFL       : {CFLmax*dt:.3f}\n"
                f"         Acoustic  : {CFLmaxA*dt:.3f}\n"
                f"         Convective: {CFLmaxC*dt:.3f}\n"
                " >>> -------------------------------- <<<\n",
            )

        mb.step(dt)

        # Check if we need to write a restart
        if mb.nrt % niterRestart == 0:
            if rank == 0:
                print("Saving restart.\n")
            pg.writers.parallelWriter.parallelWriteRestart(
                mb,
                path=config["io"]["restartDir"],
                animate=config["simulation"]["animateRestart"],
                precision="double",
            )
        # Check if we need to write archive
        if mb.nrt % niterArchive == 0:
            if rank == 0:
                print("Saving archive.\n")
            pg.writers.parallelWriter.parallelWriteRestart(
                mb,
                path=config["io"]["archiveDir"],
                animate=config["simulation"]["animateArchive"],
                precision="single",
            )

        # Check if we need to check for Nan
        if checkNan:
            if mb.nrt % checkNan == 0:
                abort = pg.mpiComm.mpiUtils.checkForNan(mb)
                if abort > 0:
                    mb.nrt = 99999999
                    pg.writers.parallelWriter.parallelWriteRestart(
                        mb,
                        path=config["io"]["restartDir"],
                        animate=True,
                    )
                    comm.Barrier()
                    if rank == 0:
                        print("Nan/inf detected. Aborting.")

                    pg.misc.abort(mb)

        # CoProcess
        mb.coproc(mb, mb.nrt)

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
