#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This utility interpolates a PEREGRINE restart from one grid to another.

Requires a path to folder that contains g.* and dtms.* grid and restart files respectively.
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
from peregrinepy.readers import readGrid, readRestart
from peregrinepy.writers import writeRestart
from peregrinepy.multiBlock import restart as mbr
from peregrinepy import interpolation
from peregrinepy.misc import progressBar
import yaml
import os


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Interpolate from one grid to another")
    parser.add_argument(
        "-from",
        "--fromDir",
        action="store",
        metavar="<fromDir>",
        dest="fromDir",
        default="./From",
        help="Directory containing the gv.*.h5 and q.*.h5 files to interpolate from. Default is ./From",
        type=str,
    )
    parser.add_argument(
        "-to",
        "--toDir",
        action="store",
        metavar="<toDir>",
        dest="toDir",
        default="./To",
        help="Directory containing the gv.*.h5 files to interpolate to. Default is ./To",
        type=str,
    )
    parser.add_argument(
        "-spdata",
        "--speciesData",
        required=True,
        action="store",
        metavar="<spdata>",
        dest="spdata",
        help="spdata yaml file (needed for species names)",
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
    with open(args.spdata, "r") as f:
        speciesNames = list(yaml.load(f, Loader=yaml.FullLoader).keys())
    function = args.function
    smooth = args.smooth
    verboseSearch = args.verboseSearch

    # Read in from data
    nblkFrom = len(
        [i for i in os.listdir(fromDir) if i.startswith("gv.") and i.endswith(".h5")]
    )
    mbFrom = mbr(nblkFrom, speciesNames)
    fromNrst = int(
        [i for i in os.listdir(fromDir) if i.startswith("q.")][0].strip().split(".")[1]
    )
    readGrid(mbFrom, fromDir)
    readRestart(mbFrom, fromDir, fromNrst)

    # Read in to data
    nblkTo = len(
        [i for i in os.listdir(toDir) if i.startswith("gv.") and i.endswith(".h5")]
    )
    mbTo = mbr(nblkTo, speciesNames)
    readGrid(mbTo, toDir)

    # Compute bounding blocks of each block
    boundsList = interpolation.bounds.findBounds(mbTo, mbFrom, verboseSearch)
    if [] in boundsList:
        raise ValueError(
            "ERROR: It looks like there are blocks in your to-grid that are completely outside your from-grid domain"
        )

    boundingBlocks = []
    for bounds in boundsList:
        boundingBlocks.append([mbFrom.getBlock(nblki) for nblki in bounds])

    total = mbTo.nblks
    for blkTo, bounds in zip(mbTo, boundingBlocks):
        interpolation.blocksToBlock(bounds, blkTo, function, smooth)
        progressBar(blkTo.nblki, total, f"Interpolating block {blkTo.nblki}")

    if len(speciesNames) > 1:
        mbTo.checkSpeciesSum(True)

    mbTo.tme = mbFrom.tme

    writeRestart(mbTo, toDir)
