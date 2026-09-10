#!/usr/bin/env python3
"""
A utility to group a grid's blocks so the computational load is balanced among
MPI ranks. The grouping is added to the grid file as a partition, named by the
number of ranks it is for, so a grid can carry several side by side.

Blocks are assigned whole, so the largest block sets a ceiling no assignment
can beat: the rank holding it cannot finish before it does. The report says
when that, rather than the grouping, is what binds.

"""

import numpy as np
import peregrinepy as pg
from peregrinepy.decomposition import (
    cellWeights,
    cutPath,
    edgesFromMb,
    metrics,
    partitionHierarchical,
    performCutOperations,
)


def getSortedBlockSizes(mb):
    sizes = np.empty(mb.nblks, dtype=np.int32)
    nblkis = np.empty(mb.nblks, dtype=np.int32)
    for i, blk in enumerate(mb):
        sizes[i] = (blk.ni - 1) * (blk.nj - 1) * (blk.nk - 1)
        nblkis[i] = blk.nblki

    perm = sizes.argsort()
    return list(sizes[perm]), list(nblkis[perm])


def analyzeLoad(procSizes, procGroups):
    assert len(procSizes) > 0
    assert len(procGroups) > 0

    maxBlksForProc = 0
    maxLoad = np.max(procSizes)

    for size, group in zip(procSizes, procGroups):
        maxBlksForProc = len(group) if len(group) > maxBlksForProc else maxBlksForProc

    perfectLoad = np.mean(procSizes)
    efficiency = perfectLoad / maxLoad * 100

    return efficiency, maxBlksForProc


def allBlocksAssigned(mb, procGroups):
    """Every block of the grid is owned by exactly one rank."""
    assigned = [nblki for group in procGroups for nblki in group]

    missing = sorted(set(mb.blockList) - set(assigned))
    if missing:
        print(f"Blocks assigned to no rank: {missing}")
    twice = sorted(n for n in set(assigned) if assigned.count(n) > 1)
    if twice:
        print(f"Blocks assigned to more than one rank: {twice}")

    return not missing and not twice and len(assigned) == mb.nblks


def blockCells(mb):
    return np.array([(b.ni - 1) * (b.nj - 1) * (b.nk - 1) for b in mb])


def cutToFit(mb, maxCells):
    """Cut blocks until none holds more than maxCells. A cut runs on through
    every block its plane meets, so the cheapest axis is the shortest path."""
    while True:
        sizes = blockCells(mb)
        worst = int(sizes.argmax())
        if sizes[worst] <= maxCells:
            return
        blk = mb[worst]

        best = None
        for axis, nNodes in zip("ijk", (blk.ni, blk.nj, blk.nk)):
            # every piece needs a cell, so n nodes take at most n-2 cuts
            nCuts = min(int(np.ceil(sizes[worst] / maxCells)) - 1, nNodes - 2)
            if nCuts < 1:
                continue
            cost = len(cutPath(mb, blk.nblki, axis)) * nCuts
            if best is None or cost < best[0]:
                best = (cost, axis, nCuts)
        if best is None:
            raise ValueError(
                f"block {blk.nblki} holds {sizes[worst]} cells and cannot be cut"
                f" under {maxCells}: it is only {blk.ni}x{blk.nj}x{blk.nk} nodes"
            )
        performCutOperations(mb, [[blk.nblki, best[1], best[2]]])


def loadCeiling(sizes, numProcs):
    """The best efficiency any grouping of whole blocks can reach. A rank
    holding the largest block cannot finish before it does, so once the
    largest block is bigger than an even share it, and not the grouping, is
    what limits the result."""
    return min(1.0, np.sum(sizes) / numProcs / np.max(sizes)) * 100


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Add a load balanced partition to a grid.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-gridDir",
        action="store",
        metavar="<gridDir>",
        dest="gridDir",
        default="./",
        help="Path to grid files",
        type=str,
    )
    parser.add_argument(
        "-numProcs",
        "--numberOfProcs",
        dest="numProcs",
        help="Number of ranks to balance for",
        required=True,
        type=int,
    )
    parser.add_argument(
        "-ranksPerNode",
        dest="ranksPerNode",
        help="".join(
            [
                "How many of those ranks share a node. Blocks are split over\n",
                "nodes first, so the network carries as little as possible,\n",
                "then over the ranks within a node, where a neighbor is only\n",
                "a shared memory copy away.",
            ]
        ),
        required=True,
        type=int,
    )
    parser.add_argument(
        "-granularity",
        dest="granularity",
        default=None,
        help="".join(
            [
                "Cut blocks down before placing them, so the load can be\n",
                "balanced past what whole blocks allow. The cap on a piece is\n",
                "an even share of the cells divided by this, so 1 cuts only\n",
                "what cannot fit a rank and 2 aims for two pieces per rank.\n",
                "Omit it to place the blocks uncut.",
            ]
        ),
        type=float,
    )

    args = parser.parse_args()
    gridDir = args.gridDir
    numProcs = args.numProcs
    ranksPerNode = args.ranksPerNode
    granularity = args.granularity

    if numProcs % ranksPerNode:
        raise SystemExit(
            f"-numProcs {numProcs} is not a whole number of nodes at"
            f" -ranksPerNode {ranksPerNode}."
        )
    nNodes = numProcs // ranksPerNode

    # cutting needs the coordinates, placing them does not
    mb = pg.multiBlock.grid.mbFromGrid(gridDir, justNi=granularity is None)

    if granularity is not None:
        cap = int(blockCells(mb).sum() / numProcs / granularity)
        before, beforeMax = len(mb), blockCells(mb).max()
        cutToFit(mb, cap)
        print(
            f"  cap {cap} cells per piece: {before} blocks -> {len(mb)} pieces,"
            f" largest {beforeMax} -> {blockCells(mb).max()}"
        )

    # the edges are keyed by block number, which is also the position in mb
    assert [blk.nblki for blk in mb] == list(range(len(mb)))
    weights, edges = cellWeights(mb), edgesFromMb(mb)
    assign = partitionHierarchical(weights, edges, nNodes, ranksPerNode)
    procGroups = [
        [int(n) for n in np.flatnonzero(assign == r)] for r in range(numProcs)
    ]
    procLoad = np.bincount(assign, weights=weights, minlength=numProcs)
    traffic = metrics(assign, weights, edges, numProcs, ranksPerNode)

    assert allBlocksAssigned(mb, procGroups)
    efficiency, maxBlksForProcs = analyzeLoad(procLoad, procGroups)

    pg.writers.GridWriter(mb, gridDir).writePartition(mb, procGroups, ranksPerNode)

    print(
        f"Added a {numProcs}x{ranksPerNode} partition"
        f" ({nNodes} node(s)) to {gridDir}/g.h5\n\n",
        "Results of Load Balancing:\n",
        f"Total Number of blocks = {len(mb)}\n",
        f"Total Number of cells  = {int(weights.sum())}\n\n",
        f"Maximum blocks on processor = {maxBlksForProcs}\n",
        f"Maximum load on processor = {int(procLoad.max())}\n",
        f"Eficiency = {efficiency}\n",
        f"Halo traffic of {traffic['totalEdge']} face cells:\n",
        f"  on a rank (free)      = {traffic['intraFraction']:.1f} %\n",
        f"  on a node (shared)    = {traffic['onNodeFraction']:.1f} %\n",
        f"  over the network      = {traffic['offNodeFraction']:.1f} %"
        f"  (busiest node {int(traffic['maxNodeTrunk'])} cells)\n",
    )

    ceiling = loadCeiling(weights, numProcs)
    if ceiling < 99.9:
        share = int(weights.sum() / numProcs)
        print(
            f" The largest block is {int(weights.max())} cells against an even"
            f" share of {share},\n so no placement of whole blocks can beat"
            f" {ceiling:.1f}%. Cut with -granularity to go further.\n"
        )

    if os.name == "posix" and "DISPLAY" in os.environ:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.set_xlabel("Processor")
        ax.set_ylabel("Number of Cells")
        ax.plot(procLoad, label="ncells", color="k")
        ax.set_ylim(bottom=0, top=None)

        ax1 = ax.twinx()
        ax1.set_ylabel("Number of Blocks")
        ax1.plot(np.array([len(i) for i in procGroups]), label="nBlocks", color="r")
        ax1.set_ylim(bottom=0, top=None)

        h1, la1 = ax.get_legend_handles_labels()
        h2, la2 = ax1.get_legend_handles_labels()
        ax.legend(h1 + h2, la1 + la2)
        plt.show()
