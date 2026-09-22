"""Writing a PEREGRINE grid, g.h5; what it holds is docs/files.md."""

from pathlib import Path

import numpy as np

from ..partition import BasePartitioner
from ..misc import Progress
from .baseWriter import BaseWriter


class GridWriter(BaseWriter):
    """Writes a multiBlock's coordinates, connectivity and partitions to the
    one file that is the grid."""

    def __init__(self, mb, fileName="g.h5", precision="single", quiet=True):
        fileName = Path(fileName)
        self.h5FileName = fileName.name
        self.xmfFileName = fileName.with_suffix(".xmf").name
        # a grid's xdmf sits beside the grid it points at
        self.gridFile = self.h5FileName
        super().__init__(mb, str(fileName.parent), precision, quiet)

    def write(self, mb):
        gf = self._openCollective(self.h5FileName)
        self._stamp(gf)
        gf.attrs["totalBlocks"] = self.totalBlocks

        # the file is collective, so every rank creates every block's datasets
        for nblki, (ni, nj, nk) in enumerate(self.extents):
            coordS = gf.create_group(f"coordinates_{nblki:06d}")
            for name in ("x", "y", "z"):
                coordS.create_dataset(name, shape=(nk, nj, ni), dtype=self.fdtype)

        # the writes are collective, so every rank walks every dataset
        mine = {blk.nblki: blk for blk in mb.blocks}
        with Progress(len(self.extents), self.quiet) as bar:
            for nblki, (ni, nj, nk) in enumerate(self.extents):
                coordS = gf[f"coordinates_{nblki:06d}"]
                blk = mine.get(nblki)
                nodes = blk.nodes.get() if blk is not None else None
                for c, name in enumerate(("x", "y", "z")):
                    if blk is None:
                        self._writeSlab(coordS[name])
                        continue
                    whole = np.ascontiguousarray(
                        nodes[blk.interior + (c,)].T, dtype=self.fdtype
                    )
                    slab = ((0, 0, 0), (nk, nj, ni))
                    self._writeSlab(coordS[name], whole, slab, slab)
                bar.step(f"Writing out block {nblki}")

        self._writeConnectivity(gf, mb)
        # the base grid is a partition like any other: one rank owning all of it
        self._writeTable(
            gf.create_group("partitions/1x1"),
            "rank",
            np.zeros(self.totalBlocks, dtype=np.int32),
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
        self._writeTable(group, "rank", rank)

        if any(blk.baseSlice is not None for blk in mb.blocks):
            self._writeTable(group, "cuts", BasePartitioner.cutTable(mb))
            self._writeConnectivity(group, mb)
        gf.close()

    def _writeConnectivity(self, group, mb):
        """Store every block's faces as a table per thing a face knows, each
        one (totalBlocks, 6). How a periodic face reaches its partner is the
        shape of the grid rather than anything about a case, so the transform
        is kept here with the rest of the connectivity."""
        shape = (self.totalBlocks, 6)
        # hdf5 has no None: -1 is no neighbor, an empty string is unset, and
        # a rotation of all zeros is a face that is not periodic, which no
        # real rotation can be
        neighbor = np.full(shape, -1, dtype=np.int32)
        orientation = np.zeros(shape, dtype=object)
        bcName = np.zeros(shape, dtype=object)
        periodicRotation = np.zeros(shape + (3, 3), dtype=np.float64)
        periodicTranslation = np.zeros(shape + (3,), dtype=np.float64)
        for blk, face in mb.faces():
            # a face's cell in the tables; nface counts from one
            mine = blk.nblki, face.nface - 1
            if face.neighbor is not None:
                neighbor[mine] = face.neighbor
            orientation[mine] = face.orientation or ""
            bcName[mine] = face.bcName or ""
            if face.periodicRotation is not None:
                periodicRotation[mine] = face.periodicRotation
                periodicTranslation[mine] = face.periodicTranslation

        if "connectivity" in group:
            del group["connectivity"]
        connS = group.create_group("connectivity")
        self._writeTable(connS, "neighbor", neighbor)
        self._writeTable(connS, "periodicRotation", periodicRotation)
        self._writeTable(connS, "periodicTranslation", periodicTranslation)
        for name, table in (("orientation", orientation), ("bcName", bcName)):
            # parallel hdf5 has no variable length strings, so size to the longest
            self._writeTable(connS, name, table.astype("S"))
