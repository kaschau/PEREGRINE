from .topology import topology
from ..readers import GridReader
from .gridBlock import gridBlock


class grid(topology):
    """A list of peregrinepy.multiBlock.grid objects.
    Inherits from peregrinepy.multiBlock.topology"""

    mbType = "grid"

    def _newBlock(self, nblki):
        return gridBlock(nblki)

    def __init__(self, nblks, ls=None):
        if ls is None:
            temp = [gridBlock(i) for i in range(nblks)]
            super().__init__(nblks, temp)
        else:
            super().__init__(nblks, ls)

    @classmethod
    def fromGrid(cls, path="./", *args, extentsOnly=False):
        """A multiBlock of this kind, sized from the grid file"""
        with GridReader(path) as reader:
            mb = cls(reader.totalBlocks, *args)
            if extentsOnly:
                reader.readExtents(mb)
            else:
                reader.readGrid(mb)
            reader.readConnectivity(mb)
        return mb

    def computeMetrics(self):
        for blk in self:
            blk.computeMetrics()

    def generateHalo(self):
        for blk in self:
            blk.generateHalo()
