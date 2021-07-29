from .face_slice import fs
from ..compute import cpg

def subsonic_inlet(blk,face,thermdat):

    p = blk.array['q'][:,:,:,0]
    p[fs[face]['s0_']] = 2.0*p[fs[face]['s1_']] - p[fs[face]['s2_']]

    #Update conserved
    cpg(blk,thermdat,face,'prims')
