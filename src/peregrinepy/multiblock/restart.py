# -*- coding: utf-8 -*-

""" multiblock.py

Authors:

Kyle Schau


This module holds the peregrinepy.multiblock object class that inherits from
python lists to create a list of peregrine.block object with added functionality and attributes

"""

from .grid import grid
from .restart_block import restart_block

class restart(grid):
    '''A list of peregrinepy.restart.restart_block objects. Inherits from peregrinepy.multiblock.grid
    '''
    mb_type = 'restart'
    def __init__(self, nblks, sp_names, ls=[]):

        if ls == []:
            temp = [restart_block(i,sp_names) for i in range(nblks)]
            super().__init__(nblks,temp)
        else:
            super().__init__(nblks,ls)

        self.nrt = 0
        self.tme = 0

    def init_restart_arrays(self):
        for blk in self:
            blk.init_restart_arrays()

    # We will make the nrt and tme attribues of the restart containter
    # properties with a setter so that setting the container value will
    # also set the block object values as well.
    @property
    def nrt(self):
        return self.__nrt
    @nrt.setter
    def nrt(self,val):
        self.__nrt = val
        for blk in self:
            blk.nrt = val
    @property
    def tme(self):
        return self.__tme
    @tme.setter
    def tme(self,val):
        self.__tme = val
        for blk in self:
            blk.tme = val
