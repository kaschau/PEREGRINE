# -*- coding: utf-8 -*-

""" multiblock.py

Authors:

Kyle Schau


This module holds the peregrinepy.multiblock object class that inherits from
python lists to create a list of peregrine.block object with added functionality and attributes

"""

from .restart import restart
from .solver_block import solver_block
from ..integrators import rk1,rk4

class solver(restart):
    '''A list of peregrinepy.restart.restart_block objects. Inherits from peregrinepy.multiblock.grid
    '''
    def __init__(self, nblks, config):
        ns = config['temp']['ns']
        temp = [solver_block(i,ns) for i in range(nblks)]
        super().__init__(nblks,ns,temp)
