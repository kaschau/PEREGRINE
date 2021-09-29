# -*- coding: utf-8 -*-

""" multiblock.py

Authors:

Kyle Schau


This module holds the peregrinepy.multiblock object class that inherits from
python lists to create a list of peregrine.block object with added functionality and attributes

"""

from .restart import restart
from .solverBlock import solverBlock
from ..grid import unifySolverGrid


class solver(restart):
    """A list of peregrinepy.restart.restart_block objects. Inherits from peregrinepy.multiblock.grid"""

    mbType = "solver"

    def __init__(self, nblks, spNames):
        assert type(spNames) is list, f"spNames must me a list not {type(spNames)}"

        temp = [solverBlock(i, spNames) for i in range(nblks)]
        super().__init__(nblks, spNames, temp)

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
        self.primaryAdvFlux = None
        self.secondaryAdvFlux = None
        self.switchAdvFlux = None

        self.diffFlux = None

        # Explicit chemistry is solved for in RHS,
        #  so we want to keep implicit chemistry
        #  separate
        self.expChem = None
        self.impChem = None

        # Parallel output
        self.parallelXmf = None

    def initSolverArrays(self, config):
        for blk in self:
            blk.initSolverArrays(config)

    def unifyGrid(self):
        unifySolverGrid(self)
