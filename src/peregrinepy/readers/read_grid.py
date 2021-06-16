# -*- coding: utf-8 -*-

#import kokkos
from ..compute_ import gen3Dview
import h5py
import numpy as np


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
            ni = list(f['dimensions']['nx'])[0]
            nj = list(f['dimensions']['ny'])[0]
            nk = list(f['dimensions']['nz'])[0]


            blk.ni = ni; blk.nj = nj; blk.nk = nk

            #blk.x_ = kokkos.array("x",
            #                     [ni,nj,nk],
            #                     dtype=kokkos.double,
            #                     space=space)
            blk.x = gen3Dview("x", ni+2,
                                   nj+2,
                                   nk+2)
            blk.array['x'] = mb.np.array(blk.x, copy=False)
            blk.array['x'][1:-1,
                           1:-1,
                           1:-1] = np.array(f['coordinates']['x']).reshape((ni, nj, nk))

            #blk.y = kokkos.array("y",
            #                     [ni,nj,nk],
            #                     dtype=kokkos.double,
            #                     space=space)
            blk.y = gen3Dview("y",ni+2,
                                  nj+2,
                                  nk+2)
            blk.array['y'] = mb.np.array(blk.y, copy=False)
            blk.array['y'][1:-1,
                           1:-1,
                           1:-1] = np.array(f['coordinates']['y']).reshape((ni, nj, nk))

            #blk.z = kokkos.array("z",
            #                     [ni,nj,nk],
            #                     dtype=kokkos.double,
            #                     space=space)
            blk.z = gen3Dview("z",ni+2,
                                  nj+2,
                                  nk+2)
            blk.array['z'] = mb.np.array(blk.z, copy=False)
            blk.array['z'][1:-1,
                           1:-1,
                           1:-1] = np.array(f['coordinates']['z']).reshape((ni, nj, nk))
