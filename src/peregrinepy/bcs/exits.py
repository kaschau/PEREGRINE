from .face_slice import fs
from ..compute import momentum, species, cpg

def subsonic_exit(blk,face):

    q = blk.array['q'][:,:,:,1::]
    q[fs[face]['s0_']] = 2.0*q[fs[face]['s1_']] - q[fs[face]['s2_']]

    #Update density
    cpg(blk,face,'PT')
    #Update momentum
    momentum(blk,face,'u')
    #Update species mass
    species(blk,face,'Y')
