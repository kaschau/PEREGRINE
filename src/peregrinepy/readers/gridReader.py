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

        rank = np.array(self.f[f"partitions/{size}x{ranksPerNode}/rank"])
        assert len(rank) == self.totalBlocks, (
            f"the {size} rank partition covers {len(rank)} blocks, "
            f"but this grid has {self.totalBlocks}"
        )
        return [[int(n) for n in np.flatnonzero(rank == r)] for r in range(size)]

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

            coordS = self.f[f"coordinates_{blk.nblki:06d}"]

            # stored (nk, nj, ni), so the shape is the extents backwards
            nk, nj, ni = coordS["x"].shape
            blk.ni, blk.nj, blk.nk = int(ni), int(nj), int(nk)

            if not justNi:
                blk.initGridArrays()
                for name in ("x", "y", "z"):
                    blk.array[name][readS] = coordS[name][:].T

            if mb.mbType in ["grid", "restart"]:
                progressBar(blk.nblki + 1, len(mb), f"Reading in gridBlock {blk.nblki}")

    def readConnectivity(self, mb):
        """Add the stored connectivity to the faces of the blocks in mb."""
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

        mb.totalBlocks = self.totalBlocks
