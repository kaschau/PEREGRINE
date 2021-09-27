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
    """A list of peregrinepy.restart.restart_block objects. Inherits from peregrinepy.multiblock.grid"""

    mb_type = "solver"

    def __init__(self, nblks, sp_names):
        assert type(sp_names) is list, f"sp_names must me a list not {type(sp_names)}"

        temp = [solver_block(i, sp_names) for i in range(nblks)]
        super().__init__(nblks, sp_names, temp)

        # Save the config file to the mb object
        self.config = None
        # Save the species data
        self.thtrdat = None

        #########################################
        # Consistify
        #########################################
        # We need the following in order to use
        # consisify method
        self.eos = None
        self.trans = None
        self.dqdxyz = None

        #########################################
        # RHS
        #########################################
        # We need the following in order to use
        # RHS method
        self.nonDissAdvFlx = None
        self.DissAdvFlx = None
        self.switch = None

        self.diffFlx = None

        self.chem = None

        # Parallel output
        self.parallel_xmf = None

    def init_solver_arrays(self, config):
        for blk in self:
            blk.init_solver_arrays(config)

    def unify_grid(self):
        unify_solver_grid(self)
