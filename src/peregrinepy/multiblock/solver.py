# -*- coding: utf-8 -*-

""" multiblock.py

Authors:

Kyle Schau


This module holds the peregrinepy.multiblock object class that inherits from
python lists to create a list of peregrine.block object with added functionality and attributes

"""

from .restart import restart
from .solver_block import solver_block
from ..grid import unify_solver_grid

class solver(restart):
    '''A list of peregrinepy.restart.restart_block objects. Inherits from peregrinepy.multiblock.grid
    '''
    mb_type = 'solver'
    def __init__(self, nblks, sp_names):
        assert (type(sp_names) is list), f'sp_names must me a list not {type(sp_names)}'

        temp = [solver_block(i,sp_names) for i in range(nblks)]
        super().__init__(nblks,sp_names,temp)

        self.config = None
        self.thermdat = None
        self.eos = None

        #Parallel output
        self.parallel_xmf = None

    def init_solver_arrays(self,config):
        for blk in self:
            blk.init_solver_arrays(config)

    def unify_solver_grid(self):
        unify_solver_grid(self)
