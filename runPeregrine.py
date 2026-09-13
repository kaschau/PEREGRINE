#!/usr/bin/env -S python -m mpi4py
import argparse
import sys
from time import perf_counter

import numpy as np

import peregrinepy as pg

np.seterr(all="raise")


def simulate(args):
    comm, rank, size = pg.mpiComm.mpiUtils.getCommRankSize()
    if rank == 0:
        string = " >>> ******************************** <<<\n"
        string += "              PEREGRINE CFD\n"
        string += " >>> ******************************** <<<\n"
        string += "  Copyright (c) 2021-2024 Kyle A. Schau\n"
        string += "           All rights reserved.\n"
        print(string)

    # a result carries the case and the grid it came from; either given here wins
    state = pg.readers.RestartReader(args.restart) if args.restart else None
    config = pg.readers.readConfigFile(args.config) if args.config else state.config
    ranks = (size, pg.mpiComm.mpiUtils.getRanksPerNode())
    mesh = pg.readers.GridReader(args.mesh or state.grid, ranks)
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
    parser = argparse.ArgumentParser(
        description="Run a PEREGRINE case: from a config and a grid, or from a "
        "result, which carries both."
    )
    parser.add_argument("config", nargs="?", help="the case's yaml")
    parser.add_argument("mesh", nargs="?", help="the grid file, g.h5")
    parser.add_argument("-r", "--restart", help="a result to restart from, q.<nrt>.h5")
    args = parser.parse_args()
    if not args.restart and not (args.config and args.mesh):
        parser.error("a config and a grid file, or a result to restart from")
    try:
        pg.abi.lib.initialize()
        simulate(args)
        pg.abi.lib.finalize()

    except Exception as e:
        import traceback

        print(f"{e}")
        excType, excValue, excTraceback = sys.exc_info()
        traceback.print_exception(excType, excValue, excTraceback)
        sys.exit(1)
