"""
Writing a PEREGRINE grid.

g.h5 carries the grid and everything about it that does not change with the
case: the coordinates of every block, how the blocks connect to each other,
and any partitions the grid has been balanced into.

    g.h5
      totalBlocks                                        attribute
      coordinates_000000/{x,y,z}                         one group per block
      connectivity/{neighbor,orientation,bcType,bcFam}    (totalBlocks, 6)
      partitions/1x1/rank                                 the base grid
      partitions/64x4/rank                                which rank owns each

Coordinates are stored (nk, nj, ni), which is the order the device already
holds them in -- the views are LayoutLeft on GPU, so i is fastest and the
transpose to this shape is free. It also makes a block's extents the shape of
its datasets rather than a second thing that can disagree with them.

A grid carries as many partitions side by side as it has been balanced for,
named ranks x ranksPerNode, so one grid runs on 64 ranks of 4 per node or of
8 without being rebalanced -- the two place blocks differently, because what
crosses a node costs more than what stays on one. Partition 1x1 is the base
grid, every block on one rank, and is written with the grid so it is never a
special case. Boundary condition values stay in the case's bcFams.yaml -- they
belong to the case, not to the grid.
"""

import numpy as np

from ..decomposition import cutTable
from .baseWriter import BaseWriter


class GridWriter(BaseWriter):
    """Writes a multiBlock's coordinates, connectivity and partitions to the
    one file that is the grid."""

    def __init__(self, mb, path="./", precision="single"):
        # a grid's xdmf sits beside the grid it points at
        self.gridPath = "."
        super().__init__(mb, path, precision)

    @property
    def h5FileName(self):
        return "g.h5"

    @property
    def xmfFileName(self):
        return "g.xmf"

    def write(self, mb):
        gf = self._openCollective(self.h5FileName)
        gf.attrs["totalBlocks"] = self.totalBlocks

        # the file is collective, so every rank creates every block's datasets
        for nblki, (ni, nj, nk) in enumerate(self.extents):
            coordS = gf.create_group(f"coordinates_{nblki:06d}")
            for name in ("x", "y", "z"):
                coordS.create_dataset(name, shape=(nk, nj, ni), dtype=self.fdtype)

        for blk in mb:
            coordS = gf[f"coordinates_{blk.nblki:06d}"]
            for name in ("x", "y", "z"):
                coordS[name][:] = np.ascontiguousarray(blk.array[name][blk.interior].T)
            mb.progress(blk.nblki + 1, f"Writing out gridBlock {blk.nblki}")

        self._writeConnectivity(gf, mb)
        # the base grid is a partition like any other: one rank owning all of it
        gf.create_dataset(
            "partitions/1x1/rank", data=np.zeros(self.totalBlocks, dtype=np.int32)
        )
        gf.close()

        self.saveXdmf()

    def writePartition(self, mb, blocksForProcs, ranksPerNode):
        """Add a partition of this grid's blocks to its grid file, named by
        the number of ranks it is for. A grid keeps every partition it has
        been balanced into, so one grid runs on any of them. Replaces any
        partition the grid already carries for that many ranks.

        A partition whose blocks are pieces of the base grid also stores the
        cut table that says which slab of which base block each piece is, and
        the pieces' own connectivity. One whose blocks are the base blocks
        needs neither, and uses the grid's.

        Parameters
        ----------
        mb : peregrinepy.multiBlock.grid (or a descendant)
            The blocks being partitioned, base blocks or pieces of them

        blocksForProcs : list
            List of lists, the first index the rank, the second its block
            number(s)

        ranksPerNode : int
            How many of those ranks share a node, which is what the placement
            was optimized for

        Returns
        -------
        None

        """

        name = f"{len(blocksForProcs)}x{ranksPerNode}"
        rank = np.full(self.totalBlocks, -1, dtype=np.int32)
        for r, group in enumerate(blocksForProcs):
            for nblki in group:
                rank[nblki] = r
        assert not (rank == -1).any(), "every block must be owned by a rank"

        gf = self._openCollective(self.h5FileName, mode="a")
        if f"partitions/{name}" in gf:
            del gf[f"partitions/{name}"]
        group = gf.create_group(f"partitions/{name}")
        group.create_dataset("rank", data=rank)

        if any(blk.baseSlice is not None for blk in mb):
            group.create_dataset("cuts", data=cutTable(mb))
            self._writeConnectivity(group, mb)
        gf.close()

    def _writeConnectivity(self, group, mb):
        """Store every block's faces as four (totalBlocks, 6) tables."""
        shape = (self.totalBlocks, 6)
        # hdf5 has no None: -1 is no neighbor, an empty string is unset
        neighbor = np.full(shape, -1, dtype=np.int32)
        orientation = np.zeros(shape, dtype=object)
        bcType = np.zeros(shape, dtype=object)
        bcFam = np.zeros(shape, dtype=object)
        for blk in mb:
            for face in blk.faces:
                # a face's cell in the tables; nface counts from one
                mine = blk.nblki, face.nface - 1
                if face.neighbor is not None:
                    neighbor[mine] = face.neighbor
                orientation[mine] = face.orientation or ""
                bcType[mine] = face.bcType
                bcFam[mine] = face.bcFam or ""

        if "connectivity" in group:
            del group["connectivity"]
        connS = group.create_group("connectivity")
        connS.create_dataset("neighbor", data=neighbor)
        for name, table in (
            ("orientation", orientation),
            ("bcType", bcType),
            ("bcFam", bcFam),
        ):
            # parallel hdf5 has no variable length strings, so size to the longest
            table = table.astype("S")
            connS.create_dataset(name, data=table, dtype=table.dtype)
