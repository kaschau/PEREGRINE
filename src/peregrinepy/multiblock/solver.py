# -*- coding: utf-8 -*-

""" multiblock.py

Authors:

Kyle Schau


This module holds the peregrinepy.multiblock object class that inherits from
python lists to create a list of peregrine.block object with added functionality and attributes

"""

from .restart import restart
from .solver_block import solver_block
import cantera as ct

class solver(restart):
    '''A list of peregrinepy.restart.restart_block objects. Inherits from peregrinepy.multiblock.grid
    '''
    def __init__(self, nblks, config):

        gas = ct.Solution(config['thermochem']['ctfile'])
        ns = gas.n_species
        temp = [solver_block(i,ns) for i in range(nblks)]
        super().__init__(nblks,ns,temp)
