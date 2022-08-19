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
    comm.Barrier()
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
    niterRestart = config["io"]["niterRestart"]
    niterArchive = config["io"]["niterArchive"]
    niterPrint = config["io"]["niterPrint"]
    checkNan = config["simulation"]["checkNan"]
    for niter in range(niter):
        dt, CFLmaxA, CFLmaxC, CFLmax = pg.mpiComm.mpiUtils.getDtMaxCFL(mb)
        mb.config["timeIntegration"]["dt"] = dt
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
                mb.restartMetaData,
                path=config["io"]["restartDir"],
            )
            if mb.config["timeIntegration"]["integrator"] == "dualTime":
                pg.writers.writeDualTimeQnm1(
                    mb,
                    path=config["io"]["restartDir"],
                    animate=config["io"]["animateRestart"],
                )
        # Check if we need to write archive
        if mb.nrt % niterArchive == 0:
            if rank == 0:
                print("Saving archive.\n")
            pg.writers.parallelWriter.parallelWriteRestart(
                mb,
                mb.archiveMetaData,
                path=config["io"]["archiveDir"],
            )
            for metaData in mb.extraMetaData:
                pg.writers.parallelWriter.parallelWriteArbitraryArray(
                    mb,
                    metaData,
                    path=config["io"]["archiveDir"],
                )

        # Check if we need to check for Nan
        if checkNan:
            if mb.nrt % checkNan == 0:
                abort = pg.mpiComm.mpiUtils.checkForNan(mb)
                if abort > 0:
                    mb.nrt = 99999999
                    mb.restartMetaData.animate = True
                    mb.restartMetaData.precision = "single"
                    pg.writers.parallelWriter.parallelWriteRestart(
                        mb,
                        mb.restartMetaData,
                        path=config["io"]["restartDir"],
                    )
                    comm.Barrier()
                    if rank == 0:
                        print("Nan/inf detected. Aborting.")
                    break

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
