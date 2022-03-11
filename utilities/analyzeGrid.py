#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""This utility goes through a grid block by block and provides some statistics about the load balancing
of the grid.

"""

import os
import argparse
from peregrinepy.readers import readGrid
from peregrinepy.multiBlock import grid as mbg
import numpy as np


def analyzeGrid(mb):
    size = np.zeros(mb.nblks, dtype=np.int32)
    for blk in mb:
        size[blk.nblki] = (blk.ni - 1) * (blk.nj - 1) * (blk.nk - 1)

    assert np.min(size) > 0

    maxBlk = np.argmax(size)
    minBlk = np.argmin(size)
    bigBlk = mb.getBlock(maxBlk)
    smallBlk = mb.getBlock(minBlk)
    results = {}
    results["totalCells"] = np.sum(size)
    results["maxNblki"] = maxBlk
    results["maxCells"] = np.max(size)
    results["maxNx"] = [bigBlk.ni, bigBlk.nj, bigBlk.nk]
    results["minNblki"] = minBlk
    results["minCells"] = np.min(size)
    results["minNx"] = [smallBlk.ni, smallBlk.nj, smallBlk.nk]
    results["mean"] = np.mean(size)
    results["stdv"] = np.std(size)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze grid partition.")
    parser.add_argument(
        "-gridDir",
        action="store",
        metavar="<gridDir>",
        dest="gridDir",
        default="./",
        help="Path to grid files",
        type=str,
    )

    args = parser.parse_args()
    gp = args.gridDir

    nblks = len([i for i in os.listdir(gp) if i.startswith("g.") and i.endswith(".h5")])

    mb = mbg(nblks)
    readGrid(mb, gp)

    results = analyzeGrid(mb)

    maxNblki = results["maxNblki"]
    maxCells = results["maxCells"]
    minNblki = results["minNblki"]
    minCells = results["minCells"]
    mean = results["mean"]
    stdv = results["stdv"]

    ni, nj, nk = results["maxNx"]
    print(f"Total cells: {results['totalCells']}")
    print(f"max block is {maxNblki} with {maxCells} cells, {ni = }, {nj = }, {nk = }.")
    ni, nj, nk = results["minNx"]
    print(f"min block is {minNblki} with {minCells} cells, {ni = }, {nj = }, {nk = }.")
    print(f"{mean = }, {stdv = }")
