from ..compute_ import gen3Dview

def init_grid_arrays(mb,config):

    for blk in mb:
#-------------------------------------------------------------------------------#
#       Cell center coordinates
#-------------------------------------------------------------------------------#
        blk.xc = gen3Dview("xc", blk.ni+1,
                                 blk.nj+1,
                                 blk.nk+1)
        blk.array['xc'] = mb.np.array(blk.xc, copy=False)

        blk.yc = gen3Dview("yc", blk.ni+1,
                                 blk.nj+1,
                                 blk.nk+1)
        blk.array['yc'] = mb.np.array(blk.yc, copy=False)

        blk.zc = gen3Dview("zc", blk.ni+1,
                                 blk.nj+1,
                                 blk.nk+1)
        blk.array['zc'] = mb.np.array(blk.zc, copy=False)

#-------------------------------------------------------------------------------#
#       i face vector components and areas
#-------------------------------------------------------------------------------#
        blk.isx = gen3Dview("isx", blk.ni+2,
                                   blk.nj+1,
                                   blk.nk+1)
        blk.array['isx'] = mb.np.array(blk.isx, copy=False)

        blk.isy = gen3Dview("isy", blk.ni+2,
                                   blk.nj+1,
                                   blk.nk+1)
        blk.array['isy'] = mb.np.array(blk.isy, copy=False)

        blk.isz = gen3Dview("isz", blk.ni+2,
                                   blk.nj+1,
                                   blk.nk+1)
        blk.array['isz'] = mb.np.array(blk.isz, copy=False)

        blk.iS  = gen3Dview("iS" , blk.ni+2,
                                   blk.nj+1,
                                   blk.nk+1)
        blk.array['iS'] = mb.np.array(blk.iS, copy=False)

#-------------------------------------------------------------------------------#
#       j face vector components and areas
#-------------------------------------------------------------------------------#
        blk.jsx = gen3Dview("jsx", blk.ni+1,
                                   blk.nj+2,
                                   blk.nk+1)
        blk.array['jsx'] = mb.np.array(blk.jsx, copy=False)

        blk.jsy = gen3Dview("jsy", blk.ni+1,
                                   blk.nj+2,
                                   blk.nk+1)
        blk.array['jsy'] = mb.np.array(blk.jsy, copy=False)

        blk.jsz = gen3Dview("jsz", blk.ni+1,
                                   blk.nj+2,
                                   blk.nk+1)
        blk.array['jsz'] = mb.np.array(blk.jsz, copy=False)

        blk.jS  = gen3Dview("jS" , blk.ni+1,
                                   blk.nj+2,
                                   blk.nk+1)
        blk.array['jS'] = mb.np.array(blk.jS, copy=False)

#-------------------------------------------------------------------------------#
#       k face vector components and areas
#-------------------------------------------------------------------------------#
        blk.ksx = gen3Dview("ksx", blk.ni+1,
                                   blk.nj+1,
                                   blk.nk+2)
        blk.array['ksx'] = mb.np.array(blk.ksx, copy=False)

        blk.ksy = gen3Dview("ksy", blk.ni+1,
                                   blk.nj+1,
                                   blk.nk+2)
        blk.array['ksy'] = mb.np.array(blk.ksy, copy=False)

        blk.ksz = gen3Dview("ksz", blk.ni+1,
                                   blk.nj+1,
                                   blk.nk+2)
        blk.array['ksz'] = mb.np.array(blk.ksz, copy=False)

        blk.kS  = gen3Dview("kS" , blk.ni+1,
                                   blk.nj+1,
                                   blk.nk+2)
        blk.array['kS'] = mb.np.array(blk.kS, copy=False)

#-------------------------------------------------------------------------------#
#       k face vector components and areas
#-------------------------------------------------------------------------------#
