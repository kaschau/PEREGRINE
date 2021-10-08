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
import os


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Interpolate from one grid to another")
    parser.add_argument(
        "-from",
        "--from_dir",
        action="store",
        metavar="<from_dir>",
        dest="from_dir",
        default="./From",
        help="Directory containing the gv.*.h5 and q.*.h5 files to interpolate from. Default is ./From",
        type=str,
    )
    parser.add_argument(
        "-to",
        "--to_dir",
        action="store",
        metavar="<to_dir>",
        dest="to_dir",
        default="./To",
        help="Directory containing the gv.*.h5 files to interpolate to. Default is ./To",
        type=str,
    )
    parser.add_argument(
        "-ns",
        "--ns",
        required=True,
        action="store",
        metavar="<number_of_sepecies>",
        dest="ns",
        help="Number of species in case",
        type=int,
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
        dest="verbose_search",
        help="If on, search will explicitly go through each block and be much slower, but the interpolation quality may improve expecially if you have a lot of curvy blocks.",
    )

    args = parser.parse_args()

    from_dir = args.from_dir
    to_dir = args.to_dir
    ns = args.ns
    function = args.function
    smooth = args.smooth
    verbose_search = args.verbose_search

    # Read in from data
    nblk_from = len([i for i in os.listdir(from_dir) if i.startswith("gv.")])
    mb_from = mbr(nblk_from, ns)
    from_nrst = int(
        [i for i in os.listdir(from_dir) if i.startswith("q.")][0].strip().split(".")[1]
    )
    readGrid(mb_from, from_dir)
    readRestart(mb_from, from_dir, from_nrst)

    # Read in to data
    nblk_to = len([i for i in os.listdir(to_dir) if i.startswith("gv.")])
    mb_to = mbr(nblk_to, ns)
    readGrid(mb_to, to_dir)

    # Compute bounding blocks of each block
    bounds_list = interpolation.bounds.find_bounds(mb_to, mb_from, verbose_search)
    if [] in bounds_list:
        raise ValueError(
            "ERROR: It looks like there are blocks in your to-grid that are completely outside your from-grid domain"
        )

    bounding_blocks = []
    for bounds in bounds_list:
        bounding_blocks.append([mb_from[nblki - 1] for nblki in bounds])

    for blk_to, bounds in zip(mb_to, bounding_blocks):
        interpolation.blocks_to_block(bounds, blk_to, function, smooth)

    if ns > 1:
        mb_to.check_species_sum(True)

    mb_to.tme = mb_from.tme
    mb_to.dtm = mb_from.dtm

    writeRestart(mb_to, to_dir)
