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
        __slots__ = list(block_.__dict__.keys()) + ['nblki,'
                                                    'connectivity',
                                                    'array']

        super().__init__()
        self.nrt = 0
        self.nblki = nblki

        ################################################################################################################
        ############## Connectivity
        ################################################################################################################
        self.connectivity = FrozenDict({'1':FrozenDict({'bc':'s1', 'neighbor':None, 'orientation':None,'comm_rank':None}),
                                        '2':FrozenDict({'bc':'s1', 'neighbor':None, 'orientation':None,'comm_rank':None}),
                                        '3':FrozenDict({'bc':'s1', 'neighbor':None, 'orientation':None,'comm_rank':None}),
                                        '4':FrozenDict({'bc':'s1', 'neighbor':None, 'orientation':None,'comm_rank':None}),
                                        '5':FrozenDict({'bc':'s1', 'neighbor':None, 'orientation':None,'comm_rank':None}),
                                        '6':FrozenDict({'bc':'s1', 'neighbor':None, 'orientation':None,'comm_rank':None})})

        for i in ['1','2','3','4','5','6']:
            self.connectivity[i]._freeze()
        self.connectivity._freeze()


        ################################################################################################################
        ############## Communication
        ################################################################################################################
        self.slice_s3 = {}
        self.slice_r3 = {}
        self.slice_s4 = {}
        self.slice_r4 = {}

        self.orient = {}
        self.sendbuffer3 = {}
        self.recvbuffer3 = {}
        self.sendbuffer4 = {}
        self.recvbuffer4 = {}

        ################################################################################################################
        ############## Data arrays
        ################################################################################################################
        # Python side data
        self.array = FrozenDict()

        # Coordinate arrays
        for d in ['x','y','z']:
            self.array[f'{d}'] = None
        # Grid metrics
        # Cell centers
        for d in ['xc','yc','zc']:
            self.array[f'{d}'] = None
        # i face area vectors
        for d in ['isx','isy','isz','iS','inx','iny','inz']:
            self.array[f'{d}'] = None
        # j face area vectors
        for d in ['jsx','jsy','jsz','jS','jnx','jny','jnz']:
            self.array[f'{d}'] = None
        # k face area vectors
        for d in ['ksx','ksy','ksz','kS','knx','kny','knz']:
            self.array[f'{d}'] = None

        # Flow variables
        # Cons, prim
        for d in ['Q','q','dQ']:
            self.array[f'{d}'] = None
        # RK stages
        for d in ['rhs0','rhs1','rhs2','rhs3']:
            self.array[f'{d}'] = None
        # Face fluxes
        for d in ['iF','jF','kF']:
            self.array[f'{d}'] = None

        self.array._freeze()

