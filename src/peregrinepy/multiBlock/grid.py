from .topology import topology
from .gridBlock import gridBlock


class grid(topology):
    """A list of peregrinepy.multiBlock.grid objects.
    Inherits from peregrinepy.multiBlock.topology"""

    mbType = "grid"

    def __init__(self, nblks, ls=None):
        if ls is None:
            temp = [gridBlock(i) for i in range(nblks)]
            super().__init__(nblks, temp)
        else:
            super().__init__(nblks, ls)

    def initGridArrays(self):
        for blk in self:
            blk.initGridArrays()

    def computeMetrics(self, xcOnly=False):
        for blk in self:
            blk.computeMetrics(xcOnly)

    def generateHalo(self):
        for blk in self:
            blk.generateHalo()
