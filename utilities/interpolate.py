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
from peregrinepy.readers import readGrid, readRestart
from peregrinepy.writers import writeRestart
from peregrinepy.multiBlock import restart as mbr
from peregrinepy import interpolation
from peregrinepy.misc import progressBar
import yaml
from lxml import etree
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
        "-spdata",
        "--speciesData",
        required=True,
        action="store",
        metavar="<spdata>",
        dest="spdata",
        help="thtr yaml file, or comma separated list of species names (in order)",
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
    if os.path.isfile(args.spdata):
        with open(args.spdata, "r") as f:
            speciesNames = list(yaml.load(f, Loader=yaml.FullLoader).keys())
    else:
        speciesNames = [item for item in args.spdata.split(",") if item != ""]

    function = args.function
    smooth = args.smooth
    verboseSearch = args.verboseSearch

    # Read in from data
    tree = etree.parse(f"{fromDir}/g.xmf")
    nblkFrom = len(tree.getroot().find("Domain").find("Grid"))
    mbFrom = mbr(nblkFrom, speciesNames)
    try:
        readGrid(mbFrom, fromDir)  # lumped
        lump = True
    except FileNotFoundError:
        readGrid(mbFrom, fromDir, lump=False)  # not lumped
        lump = False

    try:
        readRestart(mbFrom, fromDir, animate=False, lump=lump)  # not animate
        animate = False
    except FileNotFoundError:
        # Try to determint nrt for animate
        qxmf = [i for i in os.listdir(fromDir) if i.endswith("xmf")][0]
        nrt = int(qxmf.strip().split(".")[1])
        readRestart(mbFrom, fromDir, nrt=nrt, animate=True, lump=lump)
        animate = True

    # Read in to data
    tree = etree.parse(f"{toDir}/g.xmf")
    nblkTo = len(tree.getroot().find("Domain").find("Grid"))
    mbTo = mbr(nblkTo, speciesNames)
    try:
        readGrid(mbTo, toDir)
        lump = True
    except FileNotFoundError:
        readGrid(mbTo, toDir, lump=False)
        lump = False

    # Compute bounding blocks of each block
    boundsList = interpolation.bounds.findBounds(mbTo, mbFrom, verboseSearch)
    if [] in boundsList:
        raise ValueError(
            "ERROR: It looks like there are blocks in your to-grid that are completely outside your from-grid domain"
        )

    boundingBlocks = []
    for bounds in boundsList:
        boundingBlocks.append([mbFrom.getBlock(nblki) for nblki in bounds])

    nblks = mbTo.nblks
    for blkTo, bounds in zip(mbTo, boundingBlocks):
        interpolation.blocksToBlock(bounds, blkTo, function, smooth)
        progressBar(blkTo.nblki + 1, nblks, f"Interpolating block {blkTo.nblki}")

    if len(speciesNames) > 1:
        mbTo.checkSpeciesSum(True)

    mbTo.tme = mbFrom.tme
    mbTo.nrt = mbFrom.nrt

    writeRestart(mbTo, toDir, lump=lump, animate=animate)
