from ..compute_ import KokkosLocation
import kokkos

if KokkosLocation in ['OpenMP','CudaUVM','Serial','Default']:
    space = kokkos.HostSpace
else:
    raise ValueError()

def init_arrays(mb,config):

    for blk in mb:
        ccshape = [blk.ni+1,blk.nj+1,blk.nk+1]
        ifshape = [blk.ni+2,blk.nj+1,blk.nk+1]
        jfshape = [blk.ni+1,blk.nj+2,blk.nk+1]
        kfshape = [blk.ni+1,blk.nj+1,blk.nk+2]

        cQshape  = [blk.ni+1,blk.nj+1,blk.nk+1,5]
        ifQshape = [blk.ni+2,blk.nj+1,blk.nk+1,5]
        jfQshape = [blk.ni+1,blk.nj+2,blk.nk+1,5]
        kfQshape = [blk.ni+1,blk.nj+1,blk.nk+2,5]

#################################################################################
######## Grid Arrays
#################################################################################
#-------------------------------------------------------------------------------#
#       Cell center coordinates
#-------------------------------------------------------------------------------#
        shape = ccshape
        for name in ['xc', 'yc', 'zc']:
            setattr(blk,name, kokkos.array(name, shape=shape, dtype=kokkos.double, space=space, dynamic=False))
            blk.array[name] = mb.np.array(getattr(blk,name), copy=False)

#-------------------------------------------------------------------------------#
#       i face vector components and areas
#-------------------------------------------------------------------------------#
        shape = ifshape
        for name in ('isx', 'isy', 'isz', 'iS', 'inx', 'iny', 'inz'):
            setattr(blk,name, kokkos.array(name, shape=shape, dtype=kokkos.double, space=space, dynamic=False))
            blk.array[name] = mb.np.array(getattr(blk,name), copy=False)

#-------------------------------------------------------------------------------#
#       j face vector components and areas
#-------------------------------------------------------------------------------#
        shape = jfshape
        for name in ('jsx', 'jsy', 'jsz', 'jS', 'jnx', 'jny', 'jnz'):
            setattr(blk,name, kokkos.array(name, shape=shape, dtype=kokkos.double, space=space, dynamic=False))
            blk.array[name] = mb.np.array(getattr(blk,name), copy=False)

#-------------------------------------------------------------------------------#
#       k face vector components and areas
#-------------------------------------------------------------------------------#
        shape = kfshape
        for name in ('ksx', 'ksy', 'ksz', 'kS', 'knx', 'kny', 'knz'):
            setattr(blk,name, kokkos.array(name, shape=shape, dtype=kokkos.double, space=space, dynamic=False))
            blk.array[name] = mb.np.array(getattr(blk,name), copy=False)

#################################################################################
######## Flow Arrays
#################################################################################
#-------------------------------------------------------------------------------#
#       Conservative, Primative, dQ
#-------------------------------------------------------------------------------#
        shape = cQshape
        for name in ('Q', 'q', 'dQ'):
            setattr(blk,name, kokkos.array(name, shape=shape, dtype=kokkos.double, space=space, dynamic=False))
            blk.array[name] = mb.np.array(getattr(blk,name), copy=False)

#-------------------------------------------------------------------------------#
#       Fluxes
#-------------------------------------------------------------------------------#
        for shape,name in zip((ifQshape,jfQshape,kfQshape),('iF', 'jF', 'kF')):
            setattr(blk,name, kokkos.array(name, shape=shape, dtype=kokkos.double, space=space, dynamic=False))
            blk.array[name] = mb.np.array(getattr(blk,name), copy=False)
