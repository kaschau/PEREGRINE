# -*- coding: utf-8 -*-

""" grid.py

Authors:

Kyle Schau


This module holds the peregrinepy.multiBlock object class that inherits from
python lists to create a list of peregrine.block object with added functionality and attributes

"""

from .topology import topology
from .gridBlock import gridBlock


class grid(topology):
    """A list of peregrinepy.multiBlock.gridBlock objects. Inherits from peregrinepy.multiBlock.topology"""

    mbType = "grid"

    def __init__(self, nblks, ls=[]):
        if ls == []:
            temp = [gridBlock(i) for i in range(nblks)]
            super().__init__(nblks, temp)
        else:
            super().__init__(nblks, ls)

    def initGridArrays(self):
        for blk in self:
            blk.initGridArrays()

    def computeMetrics(self):
        for blk in self:
            blk.computeMetrics()

    def generateHalo(self):
        for blk in self:
            blk.generateHalo()
