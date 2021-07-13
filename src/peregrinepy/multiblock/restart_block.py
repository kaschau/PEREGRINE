# -*- coding: utf-8 -*-

""" restart_block.py

Authors:

Kyle Schau

This module defines a restart_block object that is used to compose a raptorpy.multiblock.restart (see multiblock.py).

"""

import numpy as np
from .grid_block import grid_block

class restart_block(grid_block):
    '''restart_block object holds all the information that a RAPTOR restart would need to know about a block.
    '''

    block_type = 'restart'
    def __init__(self,nblki,ns):
        super().__init__(nblki)
        if ns < 1:
            raise ValueError('Number of species must be >1')
        self.ns = ns

        ################################################################################################################
        ############## Primative Variables
        ################################################################################################################
        # Conserved variables
        for d in ['q']:
            self.array[f'{d}'] = None

        if self.block_type == 'restart':
            self.array._freeze()

