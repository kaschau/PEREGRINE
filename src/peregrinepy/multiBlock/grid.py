from .topology import topology
from .gridBlock import gridBlock
from ..readers import GridReader


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
    def mbFromGrid(cls, path="./", *args, extentsOnly=False):
        """A multiBlock of this kind, sized from the grid file at :path: and
        filled with its coordinates and its connectivity. Anything the kind
        needs beyond the block count -- a restart's species names, a solver's
        halo depth -- follows the path in the order its class declares them."""
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
