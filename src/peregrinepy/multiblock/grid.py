# -*- coding: utf-8 -*-

''' grid.py

Authors:

Kyle Schau


This module holds the peregrinepy.multiblock object class that inherits from
python lists to create a list of peregrine.block object with added functionality and attributes

'''

from .topology import topology
from .grid_block import grid_block

class grid(topology):
    '''A list of raptorpy.grid.grid_block objects. Inherits from raptorpy.multiblock.dataset
    '''
    mb_type = 'grid'
    def __init__(self, nblks, ls=[]):
        if ls == []:
            temp = [grid_block(i) for i in range(nblks)]
            super().__init__(nblks,temp)
        else:
            super().__init__(nblks,ls)

    def init_grid_arrays(self):
        for blk in self:
            blk.init_grid_arrays()
