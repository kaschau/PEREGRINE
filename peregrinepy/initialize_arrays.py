
import kokkos
import peregrine as pgc
if pgc.KokkosLocation == 'Default':
    import numpy as np
    space = kokkos.HostSpace
import peregrinepy as pgpy

def initialize_arrays(blocks):

    for blk in blocks:

        blk.Qv = kokkos.array("Qv",
                              [blk.ni,blk.nj,blk.nk,5+blk.ns-1],
                              dtype=kokkos.double,
                              space=space)
        blk.Qv_np = np.array(blk.Qv, copy=False)
        blk.Qv_np[:,:,:,:] = 0.0

        blk.T = kokkos.array("T",
                              [blk.ni,blk.nj,blk.nk],
                              dtype=kokkos.double,
                              space=space)
        blk.T_np = np.array(blk.T, copy=False)
        blk.T_np[:,:,:] = 0.0

        blk.p = kokkos.array("p",
                              [blk.ni,blk.nj,blk.nk],
                              dtype=kokkos.double,
                              space=space)
        blk.p_np = np.array(blk.p, copy=False)
        blk.p_np[:,:,:] = 0.0
