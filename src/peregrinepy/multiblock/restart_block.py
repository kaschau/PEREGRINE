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
    def __init__(self,nblki,species_names=[]):
        super().__init__(nblki)

        self.nrt = 0
        self.tme = 0

        self.species_names = species_names
        self.ns = len(species_names)
        if self.ns < 1:
            raise ValueError('Number of species must be >=1')

        ################################################################################################################
        ############## Primative Variables
        ################################################################################################################
        # Conserved variables
        for d in ['q']:
            self.array[f'{d}'] = None

        if self.block_type == 'restart':
            self.array._freeze()

    def init_restart_arrays(self):
        '''
        Create zeroed numpy arrays of correct size.
        '''

        cQshape  = (self.ni+1,self.nj+1,self.nk+1,5+self.ns-1)
        self.array['q'] = np.zeros((cQshape))

    def verify_species_sum(self, normalize=False):
        '''Function to verify that the sum of species in any cell is not greater than unity
        '''

        if self.ns > 1:
            summation = np.sum(self.qv[:,:,:,5::],axis=0)
            if (np.max(summation) - 1.0) > 1e-10:
                print('Warning! Species sum of', np.max(summation), 'found at', np.unravel_index(np.argmax(summation, axis=None), summation.shape), 'in block', self.nblki)
                if normalize:
                    self.qv[:,:,:,5::] = np.where( summation > 1.0, self.qv[:,:,:,5::]/summation, self.qv[:,:,:,5::] )
                return False
        else:
            print('You are trying to check species sum on a case where ns = 1.')
