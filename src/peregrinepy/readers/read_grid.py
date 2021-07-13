# -*- coding: utf-8 -*-

from ..compute import KokkosLocation
import kokkos
import h5py
import numpy as np

if KokkosLocation in ['OpenMP','CudaUVM','Serial','Default']:
    space = kokkos.HostSpace
else:
    raise ValueError()

def read_grid(mb,path='./'):
    ''' This function reads in all the HDF5 grid files in :path: and adds the coordinate data to a supplied raptorpy.multiblock.grid object (or one of its descendants)

    Parameters
    ----------


    Returns
    -------
    None

    '''

    for blk in mb:
        file_name = f"{path}/gv.{blk.nblki:06d}.h5"

        with h5py.File(file_name, 'r') as f:
            ni = list(f['dimensions']['ni'])[0]
            nj = list(f['dimensions']['nj'])[0]
            nk = list(f['dimensions']['nk'])[0]


            blk.ni = ni; blk.nj = nj; blk.nk = nk

            ccshape = [ni+2,nj+2,nk+2]
            for name in ('x','y','z'):
                setattr(blk,name, kokkos.array(name, shape=ccshape, dtype=kokkos.double, space=space, dynamic=False))
                blk.array[name] = np.array(getattr(blk,name), copy=False)

                blk.array[name][1:-1,
                                1:-1,
                                1:-1] = np.array(f['coordinates'][name]).reshape((ni, nj, nk))
