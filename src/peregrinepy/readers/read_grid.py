# -*- coding: utf-8 -*-

#import kokkos
from ..compute_ import gen3Dview,KokkosLocation
if KokkosLocation in ['OpenMP','CudaUVM','Default']:
    import numpy as np
else:
    raise ValueError(f'Unknown KokkosLocation {KokkosLocation}')
import h5py


def read_grid(mb,config):
    ''' This function reads in all the HDF5 grid files in :path: and adds the coordinate data to a supplied raptorpy.multiblock.grid object (or one of its descendants)

    Parameters
    ----------


    Returns
    -------
    None

    '''

    #At this point we can set the number of ghost layers
    ngls = 2

    for blk in mb:
        file_name = f"{config['io']['griddir']}/gv.{blk.nblki:06d}.h5"

        with h5py.File(file_name, 'r') as f:
            ni = list(f['dimensions']['nx'])[0]
            nj = list(f['dimensions']['ny'])[0]
            nk = list(f['dimensions']['nz'])[0]


            blk.ni = ni; blk.nj = nj; blk.nk = nk
            blk.ngls = ngls

            #blk.x_ = kokkos.array("x",
            #                     [ni,nj,nk],
            #                     dtype=kokkos.double,
            #                     space=space)
            blk.x_ = gen3Dview("x", ni+2*ngls,
                                    nj+2*ngls,
                                    nk+2*ngls)
            blk.x  = np.array(blk.x_, copy=False)
            blk.x[ngls:-ngls,
                  ngls:-ngls,
                  ngls:-ngls] = np.array(f['coordinates']['x']).reshape((ni, nj, nk))

            #blk.y = kokkos.array("y",
            #                     [ni,nj,nk],
            #                     dtype=kokkos.double,
            #                     space=space)
            blk.y_ = gen3Dview("y",ni+2*ngls,
                                   nj+2*ngls,
                                   nk+2*ngls)
            blk.y  = np.array(blk.y_, copy=False)
            blk.y[ngls:-ngls,
                  ngls:-ngls,
                  ngls:-ngls] = np.array(f['coordinates']['y']).reshape((ni, nj, nk))

            #blk.z = kokkos.array("z",
            #                     [ni,nj,nk],
            #                     dtype=kokkos.double,
            #                     space=space)
            blk.z_ = gen3Dview("z",ni+2*ngls,
                                   nj+2*ngls,
                                   nk+2*ngls)
            blk.z  = np.array(blk.z_, copy=False)
            blk.z[ngls:-ngls,
                  ngls:-ngls,
                  ngls:-ngls] = np.array(f['coordinates']['z']).reshape((ni, nj, nk))
