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
        self.partitions = (
            sorted(int(n) for n in self.f["partitions"])
            if "partitions" in self.f
            else []
        )

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def partition(self, size):
        """The blocks each of :size: ranks owns, or None if the grid carries no
        partition for that many.

        One rank owns the whole grid, so a partition for one is never stored.

        Returns a list of lists, the first index the rank, the second its
        block number(s).
        """
        if size == 1:
            return [list(range(self.totalBlocks))]

        if f"partitions/{size}" not in self.f:
            return None

        rank = np.array(self.f[f"partitions/{size}/rank"])
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

            nblkiS = f"{blk.nblki:06d}"
            coordS = "coordinates_" + nblkiS
            dimS = "dimensions_" + nblkiS

            ni = list(self.f[dimS]["ni"])[0]
            nj = list(self.f[dimS]["nj"])[0]
            nk = list(self.f[dimS]["nk"])[0]

            blk.ni = int(ni)
            blk.nj = int(nj)
            blk.nk = int(nk)

            if not justNi:
                blk.initGridArrays()
                for name in ("x", "y", "z"):
                    blk.array[name][readS] = np.array(self.f[coordS][name]).reshape(
                        (ni, nj, nk), order="F"
                    )

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
                # the face setters take python types, not numpy ones, and read
                # an empty string as unset and -1 as no neighbor
                face.bcType = str(bcType[mine])
                face.bcFam = str(bcFam[mine]) or None
                face.orientation = str(orientation[mine]) or None
                n = int(neighbor[mine])
                face.neighbor = None if n == -1 else n

        mb.totalBlocks = self.totalBlocks
