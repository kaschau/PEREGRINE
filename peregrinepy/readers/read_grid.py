# -*- coding: utf-8 -*-

import kokkos
import peregrine as pgc
if pgc.KokkosLocation == 'Default':
    import numpy as np
    space = kokkos.HostSpace
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
            ni = np.array(f['dimensions']['nx'])[0]
            nj = np.array(f['dimensions']['ny'])[0]
            nk = np.array(f['dimensions']['nz'])[0]

            blk.ni = ni; blk.nj = nj; blk.nk = nk

            blk.x = kokkos.array("x",
                                 [ni,nj,nk],
                                 dtype=kokkos.double,
                                 space=space)
            blk.x_np = np.array(blk.x, copy=False)
            blk.x_np[:,:,:] = np.array(f['coordinates']['x']).reshape((ni, nj, nk))

            blk.y = kokkos.array("y",
                                 [ni,nj,nk],
                                 dtype=kokkos.double,
                                 space=space)
            blk.y_np = np.array(blk.y, copy=False)
            blk.y_np[:,:,:] = np.array(f['coordinates']['y']).reshape((ni, nj, nk))

            blk.z = kokkos.array("z",
                                 [ni,nj,nk],
                                 dtype=kokkos.double,
                                 space=space)
            blk.z_np = np.array(blk.z, copy=False)
            blk.z_np[:,:,:] = np.array(f['coordinates']['z']).reshape((ni, nj, nk))
