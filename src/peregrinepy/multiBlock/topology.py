from ..misc import getCommRankSize
from ..readers import GridReader
from .topologyBlock import topologyBlock


class topology:
    """The connectivity of a case: its blocks and how their faces join. A
    multiBlock starts with no blocks; whoever knows how many there are -- a
    mesher, a grid file, a partition -- adds them."""

    def __init__(self):
        self.blocks = []
        self.totalBlocks = None

    @classmethod
    def fromGrid(cls, fileName, *args, quiet=True):
        """A multiBlock of this kind, filled in from the grid file. A
        topology takes only what it can hold, which is the cheapest read of
        a grid there is: no coordinate leaves the file."""
        mb = cls(*args)
        GridReader(fileName, quiet=quiet).fill(mb)
        return mb

    def faces(self):
        """Every face of every block, as (block, face)."""
        for blk in self.blocks:
            for face in blk.faces:
                yield blk, face

    def neighborFace(self, face):
        """The face a connected block face meets, when the block across it
        is on this rank; None for a boundary or a block held elsewhere."""
        if face.neighbor is None:
            return None
        blk = self.getBlock(face.neighbor)
        return blk.getFace(face.neighborNface) if blk else None

    ###########################################################################
    # The connected block faces, by where the block across them is
    ###########################################################################
    @property
    def connOnRankFaces(self):
        """The faces connected to a block on this rank."""
        return [
            f for _, f in self.faces() if f.neighbor is not None and not f.connOffRank
        ]

    @property
    def connOffRankFaces(self):
        """The faces connected to a block on another rank, whose halos come
        by message."""
        return [f for _, f in self.faces() if f.connOffRank]

    def getBlock(self, nblki):
        """The block numbered nblki, or None if this rank does not hold it. A
        rank holds a slice of the grid's blocks, so a block's number is only
        its index here when the whole grid is on one rank."""
        if nblki < len(self.blocks) and self.blocks[nblki].nblki == nblki:
            return self.blocks[nblki]
        # Otherwise manually search for it
        for blk in self.blocks:
            if blk.nblki == nblki:
                return blk

    def _newBlock(self, nblki):
        return topologyBlock(nblki)

    def addBlock(self, nblki=None):
        """A new block, numbered nblki or next in line."""
        if nblki is None:
            nblki = max((blk.nblki for blk in self.blocks), default=-1) + 1
        blk = self._newBlock(nblki)
        self.blocks.append(blk)
        return blk

    def __repr__(self):
        string = "Topology multiBlock object:\n"
        string += f"{len(self.blocks)} block(s)\n"
        return string
