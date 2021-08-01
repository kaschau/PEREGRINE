from .face_slice import fs
from ..compute import cpg

def subsonic_exit(blk,face,thermdat):

    q = blk.array['q'][:,:,:,1::]
    q[fs[face]['s0_']] = 2.0*q[fs[face]['s1_']] - q[fs[face]['s2_']]

    #Update conservatives
    cpg(blk,thermdat,face,'prims')
