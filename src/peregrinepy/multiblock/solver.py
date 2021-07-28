# -*- coding: utf-8 -*-

""" multiblock.py

Authors:

Kyle Schau


This module holds the peregrinepy.multiblock object class that inherits from
python lists to create a list of peregrine.block object with added functionality and attributes

"""

from .restart import restart
from .solver_block import solver_block

class solver(restart):
    '''A list of peregrinepy.restart.restart_block objects. Inherits from peregrinepy.multiblock.grid
    '''
    def __init__(self, nblks, sp_names):

        temp = [solver_block(i,sp_names) for i in range(nblks)]
        super().__init__(nblks,sp_names,temp)
