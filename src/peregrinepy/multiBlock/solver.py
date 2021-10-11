# -*- coding: utf-8 -*-

""" multiBlock.py

Authors:

Kyle Schau


This module holds the peregrinepy.multiBlock object class that inherits from
python lists to create a list of peregrine.block object with added functionality and attributes

"""

from .restart import restart
from .solverBlock import solverBlock
from ..grid import unifySolverGrid


class solver(restart):
    """A list of peregrinepy.restart.restart_block objects. Inherits from peregrinepy.multiBlock.grid"""

    mbType = "solver"

    def __init__(self, nblks, spNames, ng):
        assert type(spNames) is list, f"spNames must me a list not {type(spNames)}"

        temp = [solverBlock(i, spNames, ng) for i in range(nblks)]
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
