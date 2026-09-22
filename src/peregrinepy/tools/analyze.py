"""What a grid's blocks come to: the cell count, the largest and smallest
block, and the partitions the file carries."""

import numpy as np

import peregrinepy as pg

name = "analyze"
help = "report a grid's block sizes and the partitions it carries"


def addArguments(parser):
    parser.add_argument("grid", help="the grid file")


def analyzeGrid(mb):
    size = np.zeros(len(mb.blocks), dtype=np.int64)
    for blk in mb.blocks:
        size[blk.nblki] = blk.nCells
    assert np.min(size) > 0

    maxBlk, minBlk = int(np.argmax(size)), int(np.argmin(size))
    bigBlk, smallBlk = mb.getBlock(maxBlk), mb.getBlock(minBlk)
    return {
        "totalCells": int(np.sum(size)),
        "maxNblki": maxBlk,
        "maxCells": int(np.max(size)),
        "maxNx": [bigBlk.ni, bigBlk.nj, bigBlk.nk],
        "minNblki": minBlk,
        "minCells": int(np.min(size)),
        "minNx": [smallBlk.ni, smallBlk.nj, smallBlk.nk],
        "mean": float(np.mean(size)),
        "stdv": float(np.std(size)),
    }


def main(args):
    # the extents say it all, so no coordinate is read
    mb = pg.multiBlock.topology.fromGrid(args.grid, quiet=False)
    results = analyzeGrid(mb)

    ni, nj, nk = results["maxNx"]
    print(f"Total cells: {results['totalCells']}")
    print(
        f"max block is {results['maxNblki']} with {results['maxCells']} cells,"
        f" ni={ni}, nj={nj}, nk={nk}."
    )
    ni, nj, nk = results["minNx"]
    print(
        f"min block is {results['minNblki']} with {results['minCells']} cells,"
        f" ni={ni}, nj={nj}, nk={nk}."
    )
    print(f"mean = {results['mean']}, stdv = {results['stdv']}")

    partitions = pg.readers.GridReader(args.grid).partitions
    if partitions:
        print(
            "partitioned for "
            + ", ".join(f"{n}x{rpn}" for n, rpn in partitions)
            + " (ranks x ranksPerNode)."
        )
    else:
        print("no partitions stored; balance it before a multi-rank run.")
