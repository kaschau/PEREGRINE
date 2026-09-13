#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This utility interpolates a PEREGRINE restart from one grid to another.

Requires a path to folder that contains g.*.h5 and q.*.h5 grid and output files respectively.
Also must know the number of species in the case.

The utility will output the interpolated restart file in the "to" folder.

Has several options for interpolation function: nearest, linear, cubic (see scipy.interpolate.Rbf)

To make the process easier on the user, the utility will assume the number of blocks in each case (to/from)
based on the number of g.* and dtms.* files in the "to/from" folders.

Example
-------
interpolate.py --from </path/to/from-grid/and/restart> --to </path/to/to-grid/> --ns <number_of_species>

"""

import argparse
import peregrinepy as pg
from peregrinepy.multiBlock import restart as mbr
from peregrinepy.writers import RestartWriter
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interpolate from one grid to another")
    parser.add_argument(
        "-from",
        "--fromDir",
        action="store",
        metavar="<fromDir>",
        dest="fromDir",
        default="./from",
        help="Directory containing the g.*.h5 and q.*.h5 files to interpolate from. Default is ./from",
        type=str,
    )
    parser.add_argument(
        "-to",
        "--toDir",
        action="store",
        metavar="<toDir>",
        dest="toDir",
        default="./to",
        help="Directory containing the g.*.h5 files to interpolate to. Default is ./to",
        type=str,
    )
    parser.add_argument(
        "-func",
        "--function",
        action="store",
        metavar="<function>",
        dest="function",
        default="nearest",
        help="Interpolation type (nearest, linear, etc.)",
        type=str,
    )
    parser.add_argument(
        "-smooth",
        "--smooth",
        action="store",
        metavar="<smooth>",
        dest="smooth",
        default=0.5,
        help="Smoothing applied to interpolation ( larger is more smoothing ) Default = 0.5",
        type=float,
    )
    parser.add_argument(
        "-vs",
        "--verbose-search",
        action="store_true",
        dest="verboseSearch",
        help="If on, search will explicitly go through each block and be much slower, but the interpolation quality may improve expecially if you have a lot of curvy blocks.",
    )

    args = parser.parse_args()

    fromDir = args.fromDir
    toDir = args.toDir
    function = args.function
    smooth = args.smooth
    verboseSearch = args.verboseSearch

    # the newest result there; it names its own species and grid
    nrts = sorted(
        int(f.split(".")[1])
        for f in os.listdir(fromDir)
        if f.startswith("q.") and f.endswith(".h5")
    )
    if not nrts:
        raise FileNotFoundError(f"No results found in {fromDir}")
    mbFrom = mbr.fromResult(f"{fromDir}/q.{nrts[-1]:08d}.h5", quiet=False)
    mbTo = mbr.fromGrid(f"{toDir}/g.h5", mbFrom.speciesNames, quiet=False)

    pg.interpolation.getInterpolator(function, smooth, verboseSearch).interpolate(
        mbFrom, mbTo
    )

    RestartWriter(mbTo, toDir, quiet=False).write(mbTo)
