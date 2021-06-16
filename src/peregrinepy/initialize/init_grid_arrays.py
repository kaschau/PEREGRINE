from ..compute_ import gen3Dview

def init_grid_arrays(mb,config):

    for blk in mb:

        # Cell centers
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
