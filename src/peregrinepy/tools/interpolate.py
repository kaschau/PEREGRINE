"""A result interpolated onto another grid, written as a result on it."""

import os
from pathlib import Path

import peregrinepy as pg

name = "interpolate"
help = "interpolate a result onto another grid"


def addArguments(parser):
    parser.add_argument(
        "result", help="the result to interpolate from, which names its own grid"
    )
    parser.add_argument("grid", help="the grid file to interpolate onto")
    parser.add_argument("out", help="the result to write")
    parser.add_argument(
        "-function",
        default="nearest",
        help="nearest, or a radial basis kind scipy knows: linear, cubic, ...",
    )
    parser.add_argument(
        "-smooth",
        type=float,
        default=0.5,
        help="smoothing of a radial basis interpolation; larger is smoother",
    )
    parser.add_argument(
        "-verboseSearch",
        action="store_true",
        help="search every block for the ones a block lies in: slower, and "
        "surer where blocks are curved",
    )


def main(args):
    reader = pg.readers.RestartReader(args.result, quiet=False)
    mbFrom = pg.multiBlock.restart(reader.primVars)
    pg.readers.GridReader(reader.grid, quiet=False).fill(mbFrom)
    reader.fill(mbFrom)
    mbTo = pg.multiBlock.restart.fromGrid(args.grid, mbFrom.primVars, quiet=False)

    pg.interpolation.getInterpolator(
        args.function, args.smooth, args.verboseSearch
    ).interpolate(mbFrom, mbTo)

    # the result names its grid relative to itself
    out = Path(args.out)
    gridFile = os.path.relpath(args.grid, out.parent)
    pg.writers.RestartWriter(mbTo, str(out), gridFile, quiet=False).write(mbTo)
