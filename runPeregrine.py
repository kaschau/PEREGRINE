#!/usr/bin/env -S python -m mpi4py
import argparse
import sys

import numpy as np

import peregrinepy as pg

np.seterr(all="raise")


def simulate(args):
    """A case from a config and a grid, or from a result, run to the end.
    Nothing prints unless the config asks for the report plugin."""
    comm, rank, size = pg.mpiComm.mpiUtils.getCommRankSize()
    # a result carries the case and the grid it came from; either given here wins
    state = pg.readers.RestartReader(args.restart) if args.restart else None
    config = pg.readers.readConfigFile(args.config) if args.config else state.config
    ranks = (size, pg.mpiComm.mpiUtils.getRanksPerNode())
    mesh = pg.readers.GridReader(args.mesh or state.grid, ranks)
    pg.integrators.getSolver(config, mesh, state).run()


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
