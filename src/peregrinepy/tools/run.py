"""A case from a config and a grid, or from a result, run to the end."""

import traceback

import numpy as np

import peregrinepy as pg

name = "run"
help = "run a case from a config and a grid, or from a result, which carries both"


def addArguments(parser):
    parser.add_argument("config", nargs="?", help="the case's yaml")
    parser.add_argument("grid", nargs="?", help="the grid file")
    parser.add_argument("-r", "--restart", help="a result to restart from")


def simulate(args):
    """Runs the case the arguments name; nothing prints unless the config
    asks for the report plugin."""
    comm, rank, size = pg.misc.getCommRankSize()
    # a result carries the case and the grid it came from; either given here wins
    state = pg.readers.RestartReader(args.restart) if args.restart else None
    config = pg.readers.readConfigFile(args.config) if args.config else state.config
    ranks = (size, pg.misc.getRanksPerNode())
    mesh = pg.readers.GridReader(args.grid or state.grid, ranks)
    pg.multiBlock.solver(config, mesh, state).run()


def main(args):
    from mpi4py import MPI

    if not args.restart and not (args.config and args.grid):
        raise SystemExit(
            "peregrine run: a config and a grid, or a result to restart from"
        )
    np.seterr(all="raise")
    try:
        pg.backend.abi.lib.initialize()
        simulate(args)
        pg.backend.abi.lib.finalize()
    except Exception:
        # one rank's failure ends the run, not just that rank
        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
