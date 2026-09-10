from .topology import topology
from .gridBlock import gridBlock


class grid(topology):
    """A list of peregrinepy.multiBlock.grid objects.
    Inherits from peregrinepy.multiBlock.topology"""

    def _newBlock(self, nblki):
        return gridBlock(nblki)

    def __init__(self, nblks, ls=None):
        if ls is None:
            temp = [gridBlock(i) for i in range(nblks)]
            super().__init__(nblks, temp)
        else:
            super().__init__(nblks, ls)

    def _readBlocks(self, reader):
        """Where every block's nodes are, and so how big it is."""
        reader.readGrid(self)

    def computeMetrics(self):
        for blk in self:
            blk.computeMetrics()

    def generateHalo(self):
        for blk in self:
            blk.generateHalo()
