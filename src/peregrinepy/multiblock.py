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
    def __init__(self, nblks, config):
        temp = [block(i) for i in range(nblks)]
        super().__init__(temp)

        self.nrt = 0
        self.tme = 0.0

    def block_by_nblki(self,nblki):
        for blk in self:
            if blk.nblki == nblki:
                return blk
        raise ValueError(f'No block with nblki == {nblki} found.')

    def index_by_nblki(self,nblki):
        for i,b in enumerate(self):
            if b.nblki == nblki:
                break
        else:
            raise ValueError('No block with that nblki')
        return i

    @property
    def nblks(self):
        return len(self)
    @property
    def block_list(self):
        return [b.nblki for b in self]
