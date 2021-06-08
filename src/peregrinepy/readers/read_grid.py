# -*- coding: utf-8 -*-

#import kokkos
from .._compute import gen3Dview,KokkosLocation
if KokkosLocation in ['OpenMP','CudaUVM','Default']:
    import numpy as np
else:
    raise ValueError(f'Unknown KokkosLocation {KokkosLocation}')
import h5py


def read_grid(blocks):
    ''' This function reads in all the HDF5 grid files in :path: and adds the coordinate data to a supplied raptorpy.multiblock.grid object (or one of its descendants)

    Parameters
    ----------

    mb_data : raptorpy.multiblock.grid (or a descendant)

    file_path : str
        Path to find all the HDF5 grid files to be read in

    Returns
    -------
    None

    '''

    for blk in blocks:
        file_name = './Grid/gv.{:06d}.h5'.format(blk.nblki)

        with h5py.File(file_name, 'r') as f:
            ni = list(f['dimensions']['nx'])[0]
            nj = list(f['dimensions']['ny'])[0]
            nk = list(f['dimensions']['nz'])[0]


            blk.ni = ni; blk.nj = nj; blk.nk = nk

            #blk.x = kokkos.array("x",
            #                     [ni,nj,nk],
            #                     dtype=kokkos.double,
            #                     space=space)
            blk.x = gen3Dview("x",ni,nj,nk)
            blk.x_ = np.array(blk.x, copy=False)
            blk.x_[:,:,:] = np.array(f['coordinates']['x']).reshape((ni, nj, nk))

            #blk.y = kokkos.array("y",
            #                     [ni,nj,nk],
            #                     dtype=kokkos.double,
            #                     space=space)
            blk.y = gen3Dview("y",ni,nj,nk)
            blk.y_ = np.array(blk.y, copy=False)
            blk.y_[:,:,:] = np.array(f['coordinates']['y']).reshape((ni, nj, nk))

            #blk.z = kokkos.array("z",
            #                     [ni,nj,nk],
            #                     dtype=kokkos.double,
            #                     space=space)
            blk.z = gen3Dview("z",ni,nj,nk)
            blk.z_ = np.array(blk.z, copy=False)
            blk.z_[:,:,:] = np.array(f['coordinates']['z']).reshape((ni, nj, nk))
