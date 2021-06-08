# -*- coding: utf-8 -*-

""" multiblock.py

Authors:

Kyle Schau


This module holds the peregrinepy.multiblock object class that inherits from
python lists to create a list of peregrine.block object with added functionality and attributes

"""

from collections import UserList
from .block import block

class multiblock(UserList):
    '''A list of peregrinepy.block objects
    '''
    def __init__(self, nblks):
        temp = [block(i) for i in range(nblks)]
        super().__init__(temp)

    def block_by_nblki(self,nblki):
        for blk in self:
            if blk.nblki == int(nblki):
                return blk

        raise ValueError('No block with nblki == {} found.'.format(nblki))

    @property
    def nblks(self):
        return len(self)

