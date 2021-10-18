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
    __slots__ = ("config",
                 "thtrdat",
                 "eos",
                 "trans",
                 "dqdxyz",
                 "primaryAdvFlux",
                 "switch",
                 "secondaryAdvFlux",
                 "diffFlux",
                 "expChem",
                 "impChem",
                 "parallelXmf")

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
        self.switch = None

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

    def __repr__(self):
        string = "Solver multiBlock object:\n"
        string += f"  {self.totalBlocks} total blocks\n"
        string += f"  {self.thtrdat.speciesNames} species solved.\n"
        string += f"  {self.step.name} time integration.\n"
        string += f"  {self.eos.__name__} equation of state.\n"
        string += f"  Primary Advective Flux: {self.primaryAdvFlux.__name__}.\n"
        string += f"  Switching function: {self.switch.__name__}.\n"
        string += f"  Secondary Advective Flux: {self.secondaryAdvFlux.__name__}.\n\n"
        if self.config["RHS"]["diffusion"]:
            string += f"  {self.trans.__name__} transport equation.\n"
            string += f"  Spatial derivatives estimated with {self.config['RHS']['diffOrder']} order.\n"
        else:
            string += "  Diffusion terms not solved for.\n"
        if self.config["thermochem"]["chemistry"]:
            string += "  Explicit chemistry mechanism used: {self.expChem.__name__}.\n"
            string += "  Implicit chemistry mechanism used: {self.impChem.__name__}.\n"

        return string
