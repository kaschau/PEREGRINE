# -*- coding: utf-8 -*-
from .compute_ import block_
from .misc import FrozenDict
import kokkos
import numpy as np

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
        self.nblki = nblki

        ################################################################################################################
        ############## Connectivity
        ################################################################################################################
        self.connectivity = FrozenDict({'1':FrozenDict({'bc':'s1', 'neighbor':None, 'orientation':None,'comm_rank':None,'nface':None}),
                                        '2':FrozenDict({'bc':'s1', 'neighbor':None, 'orientation':None,'comm_rank':None,'nface':None}),
                                        '3':FrozenDict({'bc':'s1', 'neighbor':None, 'orientation':None,'comm_rank':None,'nface':None}),
                                        '4':FrozenDict({'bc':'s1', 'neighbor':None, 'orientation':None,'comm_rank':None,'nface':None}),
                                        '5':FrozenDict({'bc':'s1', 'neighbor':None, 'orientation':None,'comm_rank':None,'nface':None}),
                                        '6':FrozenDict({'bc':'s1', 'neighbor':None, 'orientation':None,'comm_rank':None,'nface':None})})

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
        for d in ['xc','yc','zc','J']:
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


    def init_koarrays(self,config):
        '''
        Create the Kokkos work arrays and python side numpy wrappers
        '''

        if config['Kokkos']['Space'] in ['OpenMP','Serial','Default']:
            space = kokkos.HostSpace
        else:
            raise ValueError('Are we ready for that?')

        ccshape = [self.ni+1,self.nj+1,self.nk+1]
        ifshape = [self.ni+2,self.nj+1,self.nk+1]
        jfshape = [self.ni+1,self.nj+2,self.nk+1]
        kfshape = [self.ni+1,self.nj+1,self.nk+2]

        cQshape  = [self.ni+1,self.nj+1,self.nk+1,5]
        ifQshape = [self.ni+2,self.nj+1,self.nk+1,5]
        jfQshape = [self.ni+1,self.nj+2,self.nk+1,5]
        kfQshape = [self.ni+1,self.nj+1,self.nk+2,5]

        #################################################################################
        ######## Grid Arrays
        #################################################################################
        #-------------------------------------------------------------------------------#
        #       Cell center coordinates
        #-------------------------------------------------------------------------------#
        shape = ccshape
        for name in ['xc', 'yc', 'zc','J']:
            setattr(self,name, kokkos.array(name, shape=shape, dtype=kokkos.double, space=space, dynamic=False))
            self.array[name] = np.array(getattr(self,name), copy=False)

        #-------------------------------------------------------------------------------#
        #       i face vector components and areas
        #-------------------------------------------------------------------------------#
        shape = ifshape
        for name in ('isx', 'isy', 'isz', 'iS', 'inx', 'iny', 'inz'):
            setattr(self,name, kokkos.array(name, shape=shape, dtype=kokkos.double, space=space, dynamic=False))
            self.array[name] = np.array(getattr(self,name), copy=False)

        #-------------------------------------------------------------------------------#
        #       j face vector components and areas
        #-------------------------------------------------------------------------------#
        shape = jfshape
        for name in ('jsx', 'jsy', 'jsz', 'jS', 'jnx', 'jny', 'jnz'):
            setattr(self,name, kokkos.array(name, shape=shape, dtype=kokkos.double, space=space, dynamic=False))
            self.array[name] = np.array(getattr(self,name), copy=False)

        #-------------------------------------------------------------------------------#
        #       k face vector components and areas
        #-------------------------------------------------------------------------------#
        shape = kfshape
        for name in ('ksx', 'ksy', 'ksz', 'kS', 'knx', 'kny', 'knz'):
            setattr(self,name, kokkos.array(name, shape=shape, dtype=kokkos.double, space=space, dynamic=False))
            self.array[name] = np.array(getattr(self,name), copy=False)

        #################################################################################
        ######## Flow Arrays
        #################################################################################
        #-------------------------------------------------------------------------------#
        #       Conservative, Primative, dQ
        #-------------------------------------------------------------------------------#
        shape = cQshape
        for name in ('Q', 'q', 'dQ'):
            setattr(self,name, kokkos.array(name, shape=shape, dtype=kokkos.double, space=space, dynamic=False))
            self.array[name] = np.array(getattr(self,name), copy=False)

        #-------------------------------------------------------------------------------#
        #       RK Stages
        #-------------------------------------------------------------------------------#
        if config['solver']['time_integration'] == 'rk1':
            stages = ('rhs0', 'rhs1')
        elif config['solver']['time_integration'] == 'rk4':
            stages = ('rhs0', 'rhs1', 'rhs2', 'rhs3')
        else:
            raise ValueError('Unknown time integrator')
        shape = cQshape
        for name in stages:
            setattr(self,name, kokkos.array(name, shape=shape, dtype=kokkos.double, space=space, dynamic=False))
            self.array[name] = np.array(getattr(self,name), copy=False)

        #-------------------------------------------------------------------------------#
        #       Fluxes
        #-------------------------------------------------------------------------------#
        for shape,name in zip((ifQshape,jfQshape,kfQshape),('iF', 'jF', 'kF')):
            setattr(self,name, kokkos.array(name, shape=shape, dtype=kokkos.double, space=space, dynamic=False))
            self.array[name] = np.array(getattr(self,name), copy=False)
