"""A load-balanced partition added to a grid file. Blocks are assigned
whole, so the largest block sets a ceiling no assignment can beat; the
report says when that, rather than the grouping, is what binds."""

import numpy as np

import peregrinepy as pg
from ..partition import getPartitioner

name = "partition"
help = "add a partition for so many ranks to a grid file, load balanced"


def addArguments(parser):
    parser.add_argument("grid", help="the grid file")
    parser.add_argument(
        "-ranks", type=int, required=True, help="how many ranks to balance for"
    )
    parser.add_argument(
        "-ranksPerNode",
        type=int,
        required=True,
        help="how many of those ranks share a node: blocks are split over nodes "
        "first, so the network carries as little as possible, then over the "
        "ranks within a node",
    )
    parser.add_argument(
        "-method",
        default="auto",
        help="which partitioner places the blocks: auto, greedy or metis",
    )
    parser.add_argument(
        "-haloCost",
        type=float,
        help="what a plane cell of a traded face costs a rank, in interior "
        "cells; omit it for the measured default",
    )
    parser.add_argument(
        "-granularity",
        type=float,
        help="cut blocks down before placing them: the cap on a piece is an "
        "even share of the cells divided by this, so 1 cuts only what cannot "
        "fit a rank and 2 aims for two pieces per rank; omit it to place the "
        "blocks uncut",
    )
    parser.add_argument(
        "-plot",
        action="store_true",
        help="show the load and block count on each rank when done",
    )


def main(args):
    ranks, ranksPerNode, granularity = args.ranks, args.ranksPerNode, args.granularity
    partitioner = getPartitioner(
        args.method, **({} if args.haloCost is None else {"haloCost": args.haloCost})
    )

    # balancing is all extents and connectivity, so no coordinate is read
    mb = pg.multiBlock.topology.fromGrid(args.grid, quiet=False)
    before, beforeMax = len(mb.blocks), partitioner.blockCells(mb).max()

    blocksForProcs = partitioner.partition(mb, ranks, ranksPerNode, granularity)

    if granularity is not None:
        print(
            f"  {before} blocks -> {len(mb.blocks)} pieces,"
            f" largest {beforeMax} -> {partitioner.blockCells(mb).max()}"
        )

    weights, edges = partitioner.workWeights(mb), partitioner.edgesFromMb(mb)
    assign = np.empty(len(mb.blocks), dtype=np.int64)
    for r, group in enumerate(blocksForProcs):
        assign[group] = r
    procLoad = np.bincount(assign, weights=weights, minlength=ranks)
    traffic = partitioner.metrics(assign, weights, edges, ranks, ranksPerNode)

    efficiency = procLoad.mean() / procLoad.max() * 100
    maxBlksForProcs = max(len(g) for g in blocksForProcs)
    nNodes = ranks // ranksPerNode

    pg.writers.GridWriter(mb, args.grid, quiet=False).writePartition(
        mb, blocksForProcs, ranksPerNode
    )

    print(
        f"Added a {ranks}x{ranksPerNode} partition"
        f" ({nNodes} node(s)) to {args.grid}\n\n",
        "Results of Load Balancing:\n",
        f"Total Number of blocks = {len(mb.blocks)}\n",
        f"Total Number of cells  = {int(partitioner.blockCells(mb).sum())}\n",
        f"Total work = {int(weights.sum())} cells, with the halos at"
        f" {partitioner.haloCost} a plane cell\n\n",
        f"Maximum blocks on processor = {maxBlksForProcs}\n",
        f"Maximum work on processor = {int(procLoad.max())}\n",
        f"Eficiency = {efficiency}\n",
        f"Halo traffic of {traffic['totalEdge']} face cells:\n",
        f"  on a rank (no message) = {traffic['intraFraction']:.1f} %\n",
        f"  on a node (shared)    = {traffic['onNodeFraction']:.1f} %\n",
        f"  over the network      = {traffic['offNodeFraction']:.1f} %"
        f"  (busiest node {int(traffic['maxNodeTrunk'])} cells)\n",
    )

    ceiling = partitioner.loadCeiling(weights, ranks)
    if ceiling < 99.9:
        share = int(weights.sum() / ranks)
        print(
            f" The largest block is {int(weights.max())} cells against an even"
            f" share of {share},\n so no placement of whole blocks can beat"
            f" {ceiling:.1f}%. Cut with -granularity to go further.\n"
        )

    if args.plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.set_xlabel("Processor")
        ax.set_ylabel("Number of Cells")
        ax.plot(procLoad, label="ncells", color="k")
        ax.set_ylim(bottom=0, top=None)

        ax1 = ax.twinx()
        ax1.set_ylabel("Number of Blocks")
        ax1.plot(np.array([len(g) for g in blocksForProcs]), label="nBlocks", color="r")
        ax1.set_ylim(bottom=0, top=None)

        h1, la1 = ax.get_legend_handles_labels()
        h2, la2 = ax1.get_legend_handles_labels()
        ax.legend(h1 + h2, la1 + la2)
        plt.show()
