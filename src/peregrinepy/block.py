# -*- coding: utf-8 -*-
from .compute_ import block_
from .misc import FrozenDict

''' block.py

Authors:

Kyle Schau

This module defines the top block class. This object is the most basic object that a multiblock datasets (see multiblock.py) can be composed of.

'''

class block(block_):
    '''block object is the most basic object a raptorpy.multiblock.dataset (or one of its descendants) can be.

    Attributes
    ---------
    nblki : int
        Block number (not the index, so the first block has nblki = 1)

    connectivity : dict
        The connectivity attribute holds the connectivity information for the instance of the block object. It is actually a dictionary
        of dictionaries where the keys of the outer dictionary are numbers 1-6 representing the faces of the block. Each value of the outer
        dict is yet another dictionary holding the face information 'bc': boundary condition type, 'connection': the block connected to,
        and 'orientation': the orientation of the connection (in RAPTOR format). The default for these values is 'bc':'s1', 'connection':'0',
        'orientation':'000' for all faces. NOTE: the face information is stored as strings for all entries, even 'connection'

    '''
    def __init__(self, nblki):
        __slots__ = list(block_.__dict__.keys()) + ['connectivity',
                                                   'np',
                                                   'array']

        super().__init__()
        self.nblki = nblki



        self.connectivity = FrozenDict({'1':FrozenDict({'bc':'s1', 'neighbor':None, 'orientation':'000','comm_rank':None}),
                                        '2':FrozenDict({'bc':'s1', 'neighbor':None, 'orientation':'000','comm_rank':None}),
                                        '3':FrozenDict({'bc':'s1', 'neighbor':None, 'orientation':'000','comm_rank':None}),
                                        '4':FrozenDict({'bc':'s1', 'neighbor':None, 'orientation':'000','comm_rank':None}),
                                        '5':FrozenDict({'bc':'s1', 'neighbor':None, 'orientation':'000','comm_rank':None}),
                                        '6':FrozenDict({'bc':'s1', 'neighbor':None, 'orientation':'000','comm_rank':None})})

        for i in ['1','2','3','4','5','6']:
            self.connectivity[i]._freeze()
        self.connectivity._freeze()

        ################################################################################################################
        ############## Data arrays
        ################################################################################################################
        # Python side data
        self.array = FrozenDict()

        # Coordinate arrays
        for d in ['x','y','z']:
            self.array[f'{d}'] = None

        self.array._freeze()
