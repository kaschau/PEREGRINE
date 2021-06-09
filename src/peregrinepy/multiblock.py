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
    __slots__ = ['np']

    def __init__(self, nblks, config):
        temp = [block(i) for i in range(nblks)]
        super().__init__(temp)
        for b in self:
            b.ngls = config['RunTime']['ngls']
        # Here we set the python side wrappers
        if config['Kokkos']['Space'] in ['OpenMP','CudaUVM','Default']:
            import numpy as np
        else:
            raise ImportError(f"Unknown Kokkos Space, {config['Kokkos']['Space']}")
        self.np = np

    def block_by_nblki(self,nblki):
        for blk in self:
            if blk.nblki == nblki:
                return blk
        raise ValueError(f'No block with nblki == {nblki} found.')

    @property
    def nblks(self):
        return len(self)
    @property
    def block_list(self):
        return [b.nblki for b in self]
