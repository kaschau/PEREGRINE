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

    __slots__ = (
        "config",
        "kokkosSpace" "thtrdat",
        "eos",
        "trans",
        "dqdxyz",
        "primaryAdvFlux",
        "applyPrimaryAdvFlux",
        "switch",
        "secondaryAdvFlux",
        "applySecondaryAdvFlux" "diffFlux",
        "applyDiffFlux",
        "expChem",
        "impChem",
        "parallelXmf",
    )

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
        self.commList = []
        self.eos = None
        self.trans = None
        self.dqdxyz = None

        #########################################
        # RHS
        #########################################
        # We need the following in order to use
        # RHS method
        self.primaryAdvFlux = None
        self.applyPrimaryAdvFlux = None
        self.secondaryAdvFlux = None
        self.applySecondaryAdvFlux = None
        self.switch = None

        self.diffFlux = None
        self.applyDiffFlux = None

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

    def setBlockCommunication(self):
        for blk in self:
            blk.setBlockCommunication()

    def __repr__(self):
        string = f"  Total blocks: {self.totalBlocks}\n"
        string += f"  Species: {self.thtrdat.speciesNames}\n"
        string += f"  Time Integrator: {self.step.name}\n"
        string += f"  Shock Handling: {self.config['RHS']['shockHandling']}\n"
        string += f"  Primary Advective Flux: {self.primaryAdvFlux.__name__}\n"
        string += f"  Switching Function: {self.switch.__name__}\n"
        string += f"  Secondary Advective Flux: {self.secondaryAdvFlux.__name__}\n"
        string += f"  Equation of State: {self.eos.__name__}\n"
        if self.config["RHS"]["diffusion"]:
            string += f"  Transport Equation: {self.trans.__name__}\n"
            string += f"  Spatial derivatives estimated with {self.config['RHS']['diffOrder']} order\n"
        else:
            string += "  Diffusion terms not solved for\n"
        string += f"  Subgrid Model: {self.sgs.__name__}\n"
        if self.config["thermochem"]["chemistry"]:
            string += f"  Explicit chemistry mechanism used: {self.expChem.__name__}\n"
            string += f"  Implicit chemistry mechanism used: {self.impChem.__name__}\n"

        return string
