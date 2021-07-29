from .face_slice import fs
from ..compute import momentum, species, cpg

def subsonic_inlet(blk,face):

    p = blk.array['q'][:,:,:,0]
    p[fs[face]['s0_']] = 2.0*p[fs[face]['s1_']] - p[fs[face]['s2_']]

    #Update density
    cpg(blk,face,'PT')
    #Update momentum
    momentum(blk,face,'u')
    #Update species mass
    species(blk,face,'Y')
