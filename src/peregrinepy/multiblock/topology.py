# -*- coding: utf-8 -*-

""" topology.py

Authors:

Kyle Schau


This module holds the peregrinepy.multiblock object class that inherits from
python lists to create a list of peregrine.block object with added functionality and attributes

"""

from collections import UserList
from .topologyBlock import topologyBlock


class topology(UserList):
    """A list of peregrinepy.block objects"""

    mbType = "topology"

    def __init__(self, nblks, ls=[]):
        if ls == []:
            temp = [topologyBlock(i) for i in range(nblks)]
            super().__init__(temp)
        else:
            super().__init__(ls)

        self.totalBlocks = None

    @property
    def nblks(self):
        return len(self)

    @property
    def block_list(self):
        return [b.nblki for b in self]

    def __repr__(self):
        string = "Topology multiblock object:\n"
        string += f"{self.nblks} block(s)\n"
        return string
