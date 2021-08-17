# -*- coding: utf-8 -*-

import numpy as np
import h5py


def read_restart(mb, path='./', nrt=0):
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
        variables = ['p','u','v','w','T'] + blk.species_names[0:-1]

        file_name = f'{path}/q.{nrt:08d}.{blk.nblki:06d}.h5'

        with h5py.File(file_name, 'r') as f:

            blk.nrt = f['iter']['nrt']
            blk.tme = f['iter']['tme']

            for i,var in enumerate(variables):
                blk.array['q'][:,:,:,i] = np.array(f['results'][var]).reshape((blk.nx-1, blk.ny-1, blk.nz-1))


    #Set the mb values as well
    mb.nrt = mb[0].nrt
    mb.tme = mb[0].tme
