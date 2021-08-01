# -*- coding: utf-8 -*-

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
            ni = list(f['dimensions']['ni'])[0]
            nj = list(f['dimensions']['nj'])[0]
            nk = list(f['dimensions']['nk'])[0]

            blk.ni = ni; blk.nj = nj; blk.nk = nk

            blk.init_grid_arrays()

            for name in ('x','y','z'):
                if blk.block_type == 'solver':
                    blk.array[name][1:-1,
                                    1:-1,
                                    1:-1] = np.array(f['coordinates'][name]).reshape((ni, nj, nk))
                else:
                    blk.array[name][:] = np.array(f['coordinates'][name]).reshape((ni, nj, nk))

        if blk.block_type in ['restart_block', 'solver_block']:
            blk.init_restart_arrays()
        if blk.block_type == 'solver_block':
            blk.init_solver_arrays()
