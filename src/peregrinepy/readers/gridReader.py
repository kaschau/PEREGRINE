"""
Reading a PEREGRINE grid.

One g.h5 holds the coordinates of every block, the connectivity between them,
and any partitions the grid has been balanced into. See
peregrinepy/writers/writeGrid.py for the layout. Open it once and ask it for
what you need.
"""

import h5py
import numpy as np
from ..misc import progressBar


class Partition:
    """One decomposition the grid carries: which rank owns each block, and,
    when its blocks are pieces of the base grid rather than the base blocks
    themselves, which slab of which base block each piece is."""

    def __init__(self, name, blocksForProcs, cuts):
        self.name = name
        self.blocksForProcs = blocksForProcs
        # None when the blocks are the base blocks
        self.cuts = cuts

    def __len__(self):
        return sum(len(group) for group in self.blocksForProcs)

    def setProvenance(self, mb):
        """Tell each block which slab of which base block it is, so reading
        the grid pulls its hyperslab rather than a whole base block."""
        for blk in mb:
            if self.cuts is None:
                # the blocks are the base blocks, whatever they were numbered
                # when the multiBlock was built
                blk.baseNblki, blk.baseSlice = blk.nblki, None
                continue
            baseNblki, *bounds = self.cuts[blk.nblki]
            blk.baseNblki = int(baseNblki)
            blk.baseSlice = tuple(int(b) for b in bounds)


class GridReader:
    """The grid file in :path:.

    Opening one reads everything about the grid that is not block data: how
    many blocks it holds, and the rank counts it has been partitioned for. The
    file stays open for the block reads until it is closed.
    """

    def __init__(self, path="./"):
        self.path = path
        self.f = h5py.File(f"{path}/g.h5", "r")

        self.totalBlocks = int(self.f.attrs["totalBlocks"])
        # (ranks, ranksPerNode) of every partition the grid carries
        self.partitions = (
            sorted(tuple(int(x) for x in n.split("x")) for n in self.f["partitions"])
            if "partitions" in self.f
            else []
        )

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def partition(self, size, ranksPerNode):
        """The blocks each of :size: ranks owns, placed for :ranksPerNode: of
        them sharing a node.

        A partition for the same ranks but a different node layout assigns the
        same ranks, just placed for a machine we are not on, so it is used
        with a note rather than refused.

        Returns a list of lists, the first index the rank, the second its
        block number(s).
        """
        layouts = [rpn for n, rpn in self.partitions if n == size]
        if not layouts:
            raise ValueError(
                f"this grid carries no {size} rank partition, only "
                f"{['%dx%d' % p for p in self.partitions]}. Balance it with\n"
                f"  loadBalancer.py -gridDir {self.path} -numProcs {size}"
                f" -ranksPerNode {ranksPerNode}"
            )
        if ranksPerNode not in layouts:
            print(
                f"No {size}x{ranksPerNode} partition, using "
                f"{size}x{layouts[0]}, which was placed for a different node "
                f"layout."
            )
            ranksPerNode = layouts[0]

        name = f"{size}x{ranksPerNode}"
        group = self.f[f"partitions/{name}"]
        rank = np.array(group["rank"])
        cuts = np.array(group["cuts"]) if "cuts" in group else None
        if cuts is None:
            assert len(rank) == self.totalBlocks, (
                f"the {name} partition covers {len(rank)} blocks, "
                f"but this grid has {self.totalBlocks}"
            )
        else:
            assert len(rank) == len(cuts), (
                f"the {name} partition has {len(rank)} ranks for " f"{len(cuts)} pieces"
            )
        return Partition(
            name,
            [[int(n) for n in np.flatnonzero(rank == r)] for r in range(size)],
            cuts,
        )

    def readGrid(self, mb, justNi=False):
        """Add the coordinate data to a supplied peregrinepy.multiBlock.grid
        (or one of its descendants). With justNi, read only the block extents.
        """
        if justNi:
            assert mb.mbType not in ["restart", "solver"]

        for blk in mb:
            if blk.blockType == "solver":
                ng = blk.ng
                readS = np.s_[ng:-ng, ng:-ng, ng:-ng]
            else:
                ng = 0
                readS = np.s_[:, :, :]

            coordS = self.f[f"coordinates_{blk.baseNblki:06d}"]

            if blk.baseSlice is None:
                # stored (nk, nj, ni), so the shape is the extents backwards
                nk, nj, ni = coordS["x"].shape
                sliceS = np.s_[:, :, :]
            else:
                i0, i1, j0, j1, k0, k1 = blk.baseSlice
                ni, nj, nk = i1 - i0 + 1, j1 - j0 + 1, k1 - k0 + 1
                sliceS = np.s_[k0 : k1 + 1, j0 : j1 + 1, i0 : i1 + 1]
            blk.ni, blk.nj, blk.nk = int(ni), int(nj), int(nk)

            if not justNi:
                blk.initGridArrays()
                for name in ("x", "y", "z"):
                    blk.array[name][readS] = coordS[name][sliceS].T

            if mb.mbType in ["grid", "restart"]:
                progressBar(blk.nblki + 1, len(mb), f"Reading in gridBlock {blk.nblki}")

    def readConnectivity(self, mb, partition=None):
        """Add the stored connectivity to the faces of the blocks in mb. A
        partition whose blocks are pieces connects them its own way, so it
        carries its own."""
        if partition is not None and partition.cuts is not None:
            group = self.f[f"partitions/{partition.name}/connectivity"]
        else:
            group = self.f["connectivity"]
        neighbor = np.array(group["neighbor"])
        orientation = group["orientation"].asstr()[:]
        bcType = group["bcType"].asstr()[:]
        bcFam = group["bcFam"].asstr()[:]

        for blk in mb:
            for face in blk.faces:
                mine = blk.nblki, face.nface - 1
                # the face setters take python types, not numpy ones
                face.bcType = str(bcType[mine])
                face.bcFam = str(bcFam[mine]) or None
                face.orientation = str(orientation[mine]) or None
                n = int(neighbor[mine])
                face.neighbor = None if n == -1 else n

        mb.totalBlocks = (
            self.totalBlocks
            if partition is None or partition.cuts is None
            else len(partition)
        )
