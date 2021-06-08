# -*- coding: utf-8 -*-

import numpy as np
import h5py


def read_restart(mb, config):
    ''' This function reads in all the HDF5 grid files in :path: and adds the coordinate data to a supplied raptorpy.multiblock.grid object (or one of its descendants)

    Parameters
    ----------

    mb : raptorpy.multiblock.grid (or a descendant)

    file_path : str
        Path to find all the HDF5 grid files to be read in

    Returns
    -------
    None

    '''

    for blk in mb:

        file_name = f"{config['io']['outputdir']}/restart.{blk.nrt:08d}.{blk.nblki:06d}.h5"

        with h5py.File(file_name, 'r') as f:

            blk.qv = np.zeros((max(5,5 + blk.ns - 1), blk.nz-1, blk.ny-1, blk.nx-1))
            for i in range(max(5,5 + blk.ns - 1)):
                blk.qv[i,:,:,:] = np.array(f['results']['qv_{:02d}'.format(i+1)]).reshape((blk.nz-1, blk.ny-1, blk.nx-1))

            try:
                blk.qr = np.zeros((max(3,3 + blk.ns - 1), blk.nz-1, blk.ny-1, blk.nx-1))
                for i in range(max(3.,3 + blk.ns - 1)):
                    blk.qr[i,:,:,:] = np.array(f['results']['qr_{:02d}'.format(i)]).reshape((blk.nz-1, blk.ny-1, blk.nx-1))
            except:
                pass

            try:
                blk.qh = np.zeros((max(3,3 + blk.ns - 1), blk.nz-1, blk.ny-1, blk.nx-1))
                for i in range(max(3,3 + blk.ns - 1)):
                    blk.qh[i,:,:,:] = np.array(f['results']['qh_{:02d}'.format(i)]).reshape((blk.nz-1, blk.ny-1, blk.nx-1))
            except:
                pass

