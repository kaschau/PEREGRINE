"""
Meshing a multiBlock.

A mesher describes a shape and how finely to cut it up; handing it an empty
multiBlock fills it with blocks, their coordinates, and the wiring between
their faces. The blocks of a mesh are laid out as a cubic i,j,k lattice whatever the
shape is, so the connectivity between them is the same for every kind and
only the coordinates of a single block differ.
"""

import numpy as np


class BaseMesher:
    mesherName = None
    # a mesher makes its grid: it came from no file and no partition of one
    fileName = None
    partitionName = None

    def __init__(self, mbDims=[1, 1, 1], dimsPerBlock=[10, 10, 10]):
        self.mbDims = [int(n) for n in mbDims]
        self.dimsPerBlock = list(dimsPerBlock)

    @property
    def nblks(self):
        return int(np.prod(self.mbDims))

    def fill(self, mb):
        """Make mb's blocks, and fill in their coordinates and connectivity."""
        if mb.blocks:
            raise ValueError(
                f"a mesher fills an empty multiBlock, this one already has "
                f"{len(mb.blocks)} blocks"
            )
        for _ in range(self.nblks):
            mb.addBlock()
        mb.totalBlocks = self.nblks

        mbDims = self.mbDims
        for k in range(mbDims[2]):
            for j in range(mbDims[1]):
                for i in range(mbDims[0]):
                    blkNum = k * mbDims[1] * mbDims[0] + j * mbDims[0] + i
                    blk = mb.blocks[blkNum]

                    self.shapeBlock(blk, i, j, k)
                    self.cubicConnectivity(
                        blk, mbDims, blkNum, i, j, k, *self.periodicAxes
                    )
                    # a mesher builds every block on this rank
                    for face in blk.faces:
                        face.commRank = 0 if face.neighbor is not None else None
                    self.setPeriodicFaces(blk, i, j, k)

    ###########################################################################
    # What a kind of mesh fills in
    ###########################################################################
    @property
    def periodicAxes(self):
        """Which of i, j, k wrap around, for the connectivity."""
        return (False, False, False)

    def shapeBlock(self, blk, i, j, k):
        raise NotImplementedError

    def setPeriodicFaces(self, blk, i, j, k):
        """What a wrapped face needs beyond its type."""

    ###########################################################################
    # The lattice every kind of mesh is laid out on
    ###########################################################################
    def cubicConnectivity(
        self,
        blk,
        mbDims,
        blkNum,
        i,
        j,
        k,
        periodicI=False,
        periodicJ=False,
        periodicK=False,
    ):
        # i faces
        # face 1
        face = blk.getFace(1)
        if i == 0:
            if periodicI:
                face.bcType = "periodicTrans"
                face.neighbor = blkNum + (mbDims[0] - 1)
                face.orientation = "123"
            else:
                face.bcType = "adiabaticNoSlipWall"
                face.neighbor = None
                face.orientation = None
        else:
            face.bcType = "interior"
            face.neighbor = blkNum - 1
            face.orientation = "123"

        # face 2
        face = blk.getFace(2)
        if i == mbDims[0] - 1:
            if periodicI:
                face.bcType = "periodicTrans"
                face.neighbor = blkNum - (mbDims[0] - 1)
                face.orientation = "123"
            else:
                face.bcType = "adiabaticNoSlipWall"
                face.neighbor = None
                face.orientation = None
        else:
            face.bcType = "interior"
            face.neighbor = blkNum + 1
            face.orientation = "123"

        # j faces
        # face 3
        face = blk.getFace(3)
        if j == 0:
            if periodicJ:
                face.bcType = "periodicTrans"
                face.neighbor = blkNum + mbDims[0] * (mbDims[1] - 1)
                face.orientation = "123"
            else:
                face.bcType = "adiabaticNoSlipWall"
                face.neighbor = None
                face.orientation = None
        else:
            face.bcType = "interior"
            face.neighbor = blkNum - mbDims[0]
            face.orientation = "123"

        # face 4
        face = blk.getFace(4)
        if j == mbDims[1] - 1:
            if periodicJ:
                face.bcType = "periodicTrans"
                face.neighbor = blkNum - mbDims[0] * (mbDims[1] - 1)
                face.orientation = "123"
            else:
                face.bcType = "adiabaticNoSlipWall"
                face.neighbor = None
                face.orientation = None
        else:
            face.bcType = "interior"
            face.neighbor = blkNum + mbDims[0]
            face.orientation = "123"

        # k faces
        # face 5
        face = blk.getFace(5)
        if k == 0:
            if periodicK:
                face.bcType = "periodicTrans"
                face.neighbor = blkNum + mbDims[0] * mbDims[1] * (mbDims[2] - 1)
                face.orientation = "123"
            else:
                face.bcType = "adiabaticNoSlipWall"
                face.neighbor = None
                face.orientation = None
        else:
            face.bcType = "interior"
            face.neighbor = blkNum - mbDims[0] * mbDims[1]
            face.orientation = "123"

        # face 6
        face = blk.getFace(6)
        if k == mbDims[2] - 1:
            if periodicK:
                face.bcType = "periodicTrans"
                face.neighbor = blkNum - mbDims[0] * mbDims[1] * (mbDims[2] - 1)
                face.orientation = "123"
            else:
                face.bcType = "adiabaticNoSlipWall"
                face.neighbor = None
                face.orientation = None
        else:
            face.bcType = "interior"
            face.neighbor = blkNum + mbDims[0] * mbDims[1]
            face.orientation = "123"
