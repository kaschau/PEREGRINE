#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""This utility goes through a grid block by block and provides some statistics about the load balancing
of the grid.

"""

import argparse

import numpy as np
import peregrinepy as pg
from peregrinepy.readers import GridReader


def analyzeGrid(mb):
    size = np.zeros(len(mb.blocks), dtype=np.int32)
    for blk in mb.blocks:
        size[blk.nblki] = blk.nCells

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

    mb = pg.multiBlock.grid.fromGrid(f"{gp}/g.h5", quiet=False)

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

    partitions = GridReader(f"{gp}/g.h5").partitions
    if partitions:
        print(
            "partitioned for "
            + ", ".join(f"{n}x{rpn}" for n, rpn in partitions)
            + " (ranks x ranksPerNode)."
        )
    else:
        print("no partitions stored, so a run gets one block per rank.")
