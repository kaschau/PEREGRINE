from collections import UserList

from ..misc import progressBar
from ..readers import GridReader
from .topologyBlock import topologyBlock


class topology(UserList):
    """A list of peregrinepy.multiBlock.topology block.
    Inherits from python UserList"""

    mbType = "topology"

    def __init__(self, nblks, ls=None):
        if ls is None:
            temp = [topologyBlock(i) for i in range(nblks)]
            super().__init__(temp)
        else:
            super().__init__(ls)

        self.totalBlocks = None

    @classmethod
    def fromGrid(cls, path="./", *args):
        """A multiBlock of this kind, filled in from the grid file. A
        topology takes only what it can hold, which is the cheapest read of
        a grid there is: no coordinate leaves the file."""
        with GridReader(path) as reader:
            mb = cls(reader.totalBlocks, *args)
            mb._readBlocks(reader)
            reader.readConnectivity(mb)
        return mb

    def _readBlocks(self, reader):
        """How big each block is, which is as much as a topology holds."""
        reader.readExtents(self)

    @property
    def nblks(self):
        return len(self)

    @property
    def blockList(self):
        return [b.nblki for b in self]

    def connections(self):
        """Every face that names a neighbor, as (block, face)."""
        for blk in self:
            for face in blk.faces:
                if face.neighbor is not None:
                    yield blk, face

    def boundaries(self):
        """Every face that does not, as (block, face)."""
        for blk in self:
            for face in blk.faces:
                if face.neighbor is None:
                    yield blk, face

    def progress(self, n, message):
        progressBar(n, len(self), message)

    def getBlock(self, nblki):
        """The block numbered nblki, or None if this rank does not hold it. A
        rank holds a slice of the grid's blocks, so a block's number is only
        its index here when the whole grid is on one rank."""
        if nblki < len(self) and self[nblki].nblki == nblki:
            return self[nblki]
        # Otherwise manually search for it
        for blk in self:
            if blk.nblki == nblki:
                return blk

    def _newBlock(self, nblki):
        return topologyBlock(nblki)

    def appendBlock(self):
        nblki = max((blk.nblki for blk in self), default=-1) + 1
        self.append(self._newBlock(nblki))

    def __repr__(self):
        string = "Topology multiBlock object:\n"
        string += f"{self.nblks} block(s)\n"
        return string

    # Apparently UserList borks with slices. So have to redefine here.
    def __getitem__(self, i):
        return self.data[i]
