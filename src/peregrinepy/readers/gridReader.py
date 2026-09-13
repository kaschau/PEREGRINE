"""
Reading a PEREGRINE grid.

One g.h5 holds the coordinates of every block, the connectivity between them,
and any partitions the grid has been balanced into. See
peregrinepy/writers/writeGrid.py for the layout.
"""

import h5py
import numpy as np

from ..misc import Progress
from ..mpiComm.mpiUtils import getCommRankSize


class GridReader:
    """The grid file :fileName:.

    Making one reads everything about the grid that is not block data: how
    many blocks it holds, and the rank counts it has been partitioned for.
    The file is opened again for the block reads of fill().
    """

    def __init__(self, fileName, ranks=None, quiet=True):
        """Every block of the grid, or this rank's share of the partition for
        :ranks: = (size, ranksPerNode)."""
        self.fileName = fileName
        self.quiet = quiet
        with h5py.File(self.fileName, "r") as self.f:
            self.totalBlocks = int(self.f.attrs["totalBlocks"])
            # (ranks, ranksPerNode) of every partition the grid carries
            self.partitions = (
                sorted(
                    tuple(int(x) for x in n.split("x")) for n in self.f["partitions"]
                )
                if "partitions" in self.f
                else []
            )
            # which partition we picked, and what the rest of the reads follow
            self._partitionName, self._cuts, self._rankOfNblki = None, None, None
            self.mine = (
                range(self.totalBlocks) if ranks is None else self._partition(*ranks)
            )

    def fill(self, mb):
        """Fill mb from the file: its blocks, how big each is, its coordinates
        when the block has somewhere to hold them, and the connectivity of
        their faces."""
        with h5py.File(self.fileName, "r") as self.f, Progress(
            len(self.mine), self.quiet
        ) as bar:
            for nblki in self.mine:
                blk = mb.addBlock(nblki)
                coordS, extents = self._blockBaseInfo(blk)
                blk.setExtents(*extents)
                # a topology's block is only as big as the file says; a grid's
                # takes the coordinates, a dataset each in the file and one
                # array in the block
                if "nodes" in getattr(blk, "declared", ()):
                    nodes = blk.hostCopy("nodes")
                    for c, name in enumerate(("x", "y", "z")):
                        nodes[blk.interior + (c,)] = coordS[name][blk.baseNodeSlab].T
                    blk.store("nodes", nodes)
                bar.step(f"Reading in block {nblki}")
            self._readConnectivity(mb)

    def _partition(self, size, ranksPerNode):
        """Pick the partition for :size: ranks, placed for :ranksPerNode: of
        them sharing a node, and return the blocks this rank owns. Everything
        read afterwards follows it: the coordinates each block pulls, the
        connectivity its faces get, and which rank each neighbor is on.

        A partition for the same ranks but a different node layout assigns the
        same ranks, just placed for a machine we are not on, so it is used
        with a note rather than refused.
        """
        layouts = [rpn for n, rpn in self.partitions if n == size]
        if not layouts:
            raise ValueError(
                f"this grid carries no {size} rank partition, only "
                f"{['%dx%d' % p for p in self.partitions]}. Balance it with\n"
                f"  loadBalancer.py {self.fileName} -numProcs {size}"
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
        self._partitionName, self._cuts, self._rankOfNblki = name, cuts, rank
        return [int(n) for n in np.flatnonzero(rank == getCommRankSize()[1])]

    def _blockBaseInfo(self, blk):
        """Which block of the grid this one is, which slab of it, and how big
        that slab is."""
        if self._cuts is None:
            blk.baseNblki, blk.baseSlice = blk.nblki, None
        else:
            baseNblki, *bounds = self._cuts[blk.nblki]
            blk.baseNblki = int(baseNblki)
            blk.baseSlice = tuple(int(b) for b in bounds)

        coordS = self.f[f"coordinates_{blk.baseNblki:06d}"]
        if blk.baseSlice is None:
            # stored (nk, nj, ni), so the shape is the extents backwards
            nk, nj, ni = coordS["x"].shape
        else:
            i0, i1, j0, j1, k0, k1 = blk.baseSlice
            ni, nj, nk = i1 - i0 + 1, j1 - j0 + 1, k1 - k0 + 1
        return coordS, (int(ni), int(nj), int(nk))

    @staticmethod
    def _bcTypeOf(face):
        """What a face the grid did not name is. One with a neighbor is an
        interface, and a periodic is an interface that has been moved: turned
        if its rotation is one, and only carried if it is the identity. A face
        that is named waits for the case to say what it is."""
        if face.bcName is not None:
            return face.bcType
        if face.neighbor is None:
            return "adiabaticSlipWall"
        if face.periodicRotation is None:
            return "interior"
        return (
            "periodicTrans"
            if np.allclose(face.periodicRotation, np.eye(3))
            else "periodicRot"
        )

    def _readConnectivity(self, mb):
        """Add the stored connectivity to the faces of the blocks in mb. A
        partition whose blocks are pieces connects them its own way, so it
        carries its own."""
        if self._cuts is not None:
            group = self.f[f"partitions/{self._partitionName}/connectivity"]
        else:
            group = self.f["connectivity"]
        neighbor = np.array(group["neighbor"])
        orientation = group["orientation"].asstr()[:]
        bcName = group["bcName"].asstr()[:]
        periodicRotation = np.array(group["periodicRotation"])
        periodicTranslation = np.array(group["periodicTranslation"])

        for blk, face in mb.faces():
            mine = blk.nblki, face.nface - 1
            # the connectivity is python types, not the numpy scalars
            # hdf5 hands back
            face.bcName = str(bcName[mine]) or None
            face.orientation = str(orientation[mine]) or None
            n = int(neighbor[mine])
            face.neighbor = None if n == -1 else n
            # a periodic knows how to reach its partner; a rotation of
            # all zeros is a face that is not periodic
            if periodicRotation[mine].any():
                face.setPeriodic(
                    rotation=periodicRotation[mine],
                    translation=periodicTranslation[mine],
                )
            face.bcType = self._bcTypeOf(face)
            # without a partition every block is on this rank
            if n == -1:
                face.commRank = None
            elif self._rankOfNblki is None:
                face.commRank = 0
            else:
                face.commRank = int(self._rankOfNblki[n])

        # a cut partition's blocks are its pieces, not the base grid's blocks
        mb.totalBlocks = self.totalBlocks if self._cuts is None else len(self._cuts)
