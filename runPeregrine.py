#!/usr/bin/env -S python -m mpi4py
import sys
from time import perf_counter

import numpy as np

import peregrinepy as pg

np.seterr(all="raise")


def simulate(configFilePath):
    comm, rank, size = pg.mpiComm.mpiUtils.getCommRankSize()
    if rank == 0:
        string = " >>> ******************************** <<<\n"
        string += "              PEREGRINE CFD\n"
        string += " >>> ******************************** <<<\n"
        string += "  Copyright (c) 2021-2024 Kyle A. Schau\n"
        string += "           All rights reserved.\n"
        print(string)

    config = pg.readers.readConfigFile(configFilePath, parallel=True)
    comm.Barrier()
    if rank == 0:
        print("Read config.")

    io, sim = config["io"], config["simulation"]
    ranks = (size, pg.mpiComm.mpiUtils.getRanksPerNode())
    mesh = pg.readers.GridReader(io["gridDir"], ranks, quiet=True)
    if sim["restartFrom"] is None:
        mb = pg.multiBlock.solver(config, mesh)
    else:
        state = pg.readers.RestartReader(
            io["resultsDir"], sim["restartFrom"], quiet=True
        )
        mb = pg.multiBlock.solver(config, mesh, state)

    # Get some stats about the simulation
    nCells = mb.numCells
    efficiency, slowestProc = mb.loadEfficiency
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

    mb.run()

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
        pg.abi.lib.initialize()
        simulate(configFilePath)
        pg.abi.lib.finalize()

    except Exception as e:
        import sys
        import traceback

        print(f"{e}")
        excType, excValue, excTraceback = sys.exc_info()
        traceback.print_exception(excType, excValue, excTraceback)
        sys.exit(1)
