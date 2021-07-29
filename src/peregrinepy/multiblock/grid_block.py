# -*- coding: utf-8 -*-

''' grid_block.py

Authors:

Kyle Schau

This module defines a grid_block object that is used to compose a raptorpy.multiblock.grid (or one of its descendants) (see multiblock.py)

'''

import numpy as np
from .topology_block import topology_block
from ..misc import FrozenDict


class grid_block(topology_block):
    '''grid_block object holds all the information that a grid would need to know about a block.
    '''

    block_type = 'grid'
    def __init__(self,nblki):
        super().__init__(nblki)

        self.ni = 0
        self.nj = 0
        self.nk = 0

        ################################################################################################################
        ############## Data arrays
        ################################################################################################################
        # Python side data
        self.array = FrozenDict()
        # Coordinate arrays
        for d in ['x','y','z']:
            self.array[f'{d}'] = None
        # Grid metrics
        # Cell centers
        for d in ['xc','yc','zc','J']:
            self.array[f'{d}'] = None
        # i face area vectors
        for d in ['isx','isy','isz','iS','inx','iny','inz']:
            self.array[f'{d}'] = None
        # j face area vectors
        for d in ['jsx','jsy','jsz','jS','jnx','jny','jnz']:
            self.array[f'{d}'] = None
        # k face area vectors
        for d in ['ksx','ksy','ksz','kS','knx','kny','knz']:
            self.array[f'{d}'] = None

        if self.block_type == 'grid':
            self.array._freeze()

    def init_grid_arrays(self):
        '''
        Create empty numpy arrays of correct size.
        '''

        xshape  = [self.ni+2,self.nj+2,self.nk+2]
        for name in ['x','y','z']:
            self.array[name] = np.empty((xshape))
