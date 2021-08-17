# -*- coding: utf-8 -*-

''' topology.py

Authors:

Kyle Schau


This module holds the peregrinepy.multiblock object class that inherits from
python lists to create a list of peregrine.block object with added functionality and attributes

'''

from collections import UserList
from .topology_block import topology_block

class topology(UserList):
    '''A list of peregrinepy.block objects
    '''
    mb_type = 'topology'
    def __init__(self, nblks, ls=[]):
        if ls == []:
            temp = [topology_block(i) for i in range(nblks)]
            super().__init__(temp)
        else:
            super().__init__(ls)

    @property
    def nblks(self):
        return len(self)
    @property
    def block_list(self):
        return [b.nblki for b in self]
