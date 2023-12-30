from collections import UserList
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

    @property
    def nblks(self):
        return len(self)

    @property
    def blockList(self):
        return [b.nblki for b in self]

    def getBlock(self, nblki):
        if self[nblki].nblki == nblki:
            return self[nblki]
        # Otherwise manually search for it
        for blk in self:
            if blk.nblki == nblki:
                return blk

    def appendBlock(self):
        if self.mbType in ["restart", "solver"]:
            raise TypeError("Cannot append restart or solver multiBlocks.")
        maxNblki = 0
        for blk in self:
            maxNblki = blk.nblki if blk.nblki > maxNblki else maxNblki
        tmp_blk = self[0].__class__(maxNblki + 1)

        self.append(tmp_blk)

    def __repr__(self):
        string = "Topology multiBlock object:\n"
        string += f"{self.nblks} block(s)\n"
        return string

    # Apparently UserList borks with slices. So have to redefine here.
    def __getitem__(self, i):
        return self.data[i]
