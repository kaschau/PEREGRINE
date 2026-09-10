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
from peregrinepy.partition import getPartitioner

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
        "-method",
        dest="method",
        default="auto",
        help="Which partitioner places the blocks: auto, greedy or metis.",
        type=str,
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

    partitioner = getPartitioner(args.method)

    # cutting needs the coordinates, placing them does not
    mb = pg.multiBlock.grid.fromGrid(gridDir, extentsOnly=granularity is None)
    before, beforeMax = len(mb), partitioner.blockCells(mb).max()

    blocksForProcs = partitioner.partition(mb, numProcs, ranksPerNode, granularity)

    if granularity is not None:
        print(
            f"  {before} blocks -> {len(mb)} pieces,"
            f" largest {beforeMax} -> {partitioner.blockCells(mb).max()}"
        )

    weights, edges = partitioner.cellWeights(mb), partitioner.edgesFromMb(mb)
    assign = np.empty(len(mb), dtype=np.int64)
    for r, group in enumerate(blocksForProcs):
        assign[group] = r
    procLoad = np.bincount(assign, weights=weights, minlength=numProcs)
    traffic = partitioner.metrics(assign, weights, edges, numProcs, ranksPerNode)

    efficiency = procLoad.mean() / procLoad.max() * 100
    maxBlksForProcs = max(len(g) for g in blocksForProcs)
    nNodes = numProcs // ranksPerNode

    pg.writers.GridWriter(mb, gridDir).writePartition(mb, blocksForProcs, ranksPerNode)

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

    ceiling = partitioner.loadCeiling(weights, numProcs)
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
        ax1.plot(
            np.array([len(g) for g in blocksForProcs]),
            label="nBlocks",
            color="r",
        )
        ax1.set_ylim(bottom=0, top=None)

        h1, la1 = ax.get_legend_handles_labels()
        h2, la2 = ax1.get_legend_handles_labels()
        ax.legend(h1 + h2, la1 + la2)
        plt.show()
