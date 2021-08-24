# -*- coding: utf-8 -*-
import kokkos
import numpy as np
from ..compute import block_
from .restart_block import restart_block
from ..misc import FrozenDict

''' block.py

Authors:

Kyle Schau

This module defines the top block class. This object is the most basic object that a multiblock datasets (see multiblock.py) can be composed of.

'''

class solver_block(restart_block,block_):
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

    block_type = 'solver'
    def __init__(self, nblki, sp_names):
        # The c++ stuff must be instantiated first, so that inhereted python side
        # attributes are assigned values, not defined in the upstream __init__s
        block_.__init__(self)
        restart_block.__init__(self, nblki, sp_names)

        self.ne = 5+self.ns-1

        ################################################################################################################
        ############## Solution Variables
        ################################################################################################################
        # Conserved variables
        for d in ['Q','dQ']:
            self.array[f'{d}'] = None
        # Spatial derivative of primative array
        for d in ['dqdx','dqdy','dqdz']:
            self.array[f'{d}'] = None
        # thermo,trans arrays
        for d in ['qh']:
            self.array[f'{d}'] = None
        # RK stages
        for d in ['rhs0','rhs1','rhs2','rhs3']:
            self.array[f'{d}'] = None
        # Face fluxes
        for d in ['iF','jF','kF']:
            self.array[f'{d}'] = None

        if self.block_type == 'solver':
            self.array._freeze()

    def init_solver_arrays(self,config):
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

        cQshape  = [self.ni+1,self.nj+1,self.nk+1,5+self.ns-1]
        ifQshape = [self.ni+2,self.nj+1,self.nk+1,5+self.ns-1]
        jfQshape = [self.ni+1,self.nj+2,self.nk+1,5+self.ns-1]
        kfQshape = [self.ni+1,self.nj+1,self.nk+2,5+self.ns-1]

        #################################################################################
        ######## Grid Arrays
        #################################################################################
        #-------------------------------------------------------------------------------#
        #       Primary grid coordinates
        #-------------------------------------------------------------------------------#
        shape = [self.ni+2,self.nj+2,self.nk+2]
        for name in ['x','y','z']:
            if self.array[name] is None:
                setattr(self, name, kokkos.array(name, shape=shape, dtype=kokkos.double, space=space, dynamic=False))
                self.array[name] = np.array(getattr(self, name), copy=False)
            else:
                setattr(self, name, kokkos.array(self.array[name], dtype=kokkos.double, space=space, dynamic=False))

        #-------------------------------------------------------------------------------#
        #       Cell center
        #-------------------------------------------------------------------------------#
        shape = ccshape
        for name in ('xc', 'yc', 'zc','J'):
            if self.array[name] is None:
                setattr(self, name, kokkos.array(name, shape=shape, dtype=kokkos.double, space=space, dynamic=False))
                self.array[name] = np.array(getattr(self, name), copy=False)
            else:
                setattr(self, name, kokkos.array(self.array[name], dtype=kokkos.double, space=space, dynamic=False))
        # Cell center metrics
        for name in ('dEdx','dEdy','dEdz',
                     'dNdx','dNdy','dNdz',
                     'dXdx','dXdy','dXdz'):
            if self.array[name] is None:
                setattr(self, name, kokkos.array(name, shape=shape, dtype=kokkos.double, space=space, dynamic=False))
                self.array[name] = np.array(getattr(self, name), copy=False)
            else:
                setattr(self, name, kokkos.array(self.array[name], dtype=kokkos.double, space=space, dynamic=False))

        #-------------------------------------------------------------------------------#
        #       i face vector components and areas
        #-------------------------------------------------------------------------------#
        shape = ifshape
        for name in ('isx', 'isy', 'isz', 'iS', 'inx', 'iny', 'inz'):
            if self.array[name] is None:
                setattr(self, name, kokkos.array(name, shape=shape, dtype=kokkos.double, space=space, dynamic=False))
                self.array[name] = np.array(getattr(self, name), copy=False)
            else:
                setattr(self, name, kokkos.array(self.array[name], dtype=kokkos.double, space=space, dynamic=False))

        #-------------------------------------------------------------------------------#
        #       j face vector components and areas
        #-------------------------------------------------------------------------------#
        shape = jfshape
        for name in ('jsx', 'jsy', 'jsz', 'jS', 'jnx', 'jny', 'jnz'):
            if self.array[name] is None:
                setattr(self, name, kokkos.array(name, shape=shape, dtype=kokkos.double, space=space, dynamic=False))
                self.array[name] = np.array(getattr(self, name), copy=False)
            else:
                setattr(self, name, kokkos.array(self.array[name], dtype=kokkos.double, space=space, dynamic=False))

        #-------------------------------------------------------------------------------#
        #       k face vector components and areas
        #-------------------------------------------------------------------------------#
        shape = kfshape
        for name in ('ksx', 'ksy', 'ksz', 'kS', 'knx', 'kny', 'knz'):
            if self.array[name] is None:
                setattr(self, name, kokkos.array(name, shape=shape, dtype=kokkos.double, space=space, dynamic=False))
                self.array[name] = np.array(getattr(self, name), copy=False)
            else:
                setattr(self, name, kokkos.array(self.array[name], dtype=kokkos.double, space=space, dynamic=False))

        #################################################################################
        ######## Flow Arrays
        #################################################################################
        #-------------------------------------------------------------------------------#
        #       Conservative, Primative, dQ
        #-------------------------------------------------------------------------------#
        shape = cQshape
        for name in ('Q', 'q', 'dQ'):
            if self.array[name] is None:
                setattr(self, name, kokkos.array(name, shape=shape, dtype=kokkos.double, space=space, dynamic=False))
                self.array[name] = np.array(getattr(self, name), copy=False)
            else:
                setattr(self, name, kokkos.array(self.array[name], dtype=kokkos.double, space=space, dynamic=False))

        #-------------------------------------------------------------------------------#
        #       Spatial derivative of primative array
        #-------------------------------------------------------------------------------#
        shape = cQshape
        for name in ('dqdx','dqdy','dqdz'):
            if self.array[name] is None:
                setattr(self, name, kokkos.array(name, shape=shape, dtype=kokkos.double, space=space, dynamic=False))
                self.array[name] = np.array(getattr(self, name), copy=False)
            else:
                setattr(self, name, kokkos.array(self.array[name], dtype=kokkos.double, space=space, dynamic=False))

        #-------------------------------------------------------------------------------#
        #       Thermo
        #-------------------------------------------------------------------------------#
        shape = cQshape
        for name in ['qh']:
            if self.array[name] is None:
                setattr(self, name, kokkos.array(name, shape=shape, dtype=kokkos.double, space=space, dynamic=False))
                self.array[name] = np.array(getattr(self, name), copy=False)
            else:
                setattr(self, name, kokkos.array(self.array[name], dtype=kokkos.double, space=space, dynamic=False))

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
            if self.array[name] is None:
                setattr(self, name, kokkos.array(name, shape=shape, dtype=kokkos.double, space=space, dynamic=False))
                self.array[name] = np.array(getattr(self, name), copy=False)
            else:
                setattr(self, name, kokkos.array(self.array[name], dtype=kokkos.double, space=space, dynamic=False))

        #-------------------------------------------------------------------------------#
        #       Fluxes
        #-------------------------------------------------------------------------------#
        for shape,name in zip((ifQshape,jfQshape,kfQshape),('iF', 'jF', 'kF')):
            if self.array[name] is None:
                setattr(self, name, kokkos.array(name, shape=shape, dtype=kokkos.double, space=space, dynamic=False))
                self.array[name] = np.array(getattr(self, name), copy=False)
            else:
                setattr(self, name, kokkos.array(self.array[name], dtype=kokkos.double, space=space, dynamic=False))
