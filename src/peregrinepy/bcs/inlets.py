from .face_slice import fs
from ..compute_ import momentum, EOS_ideal, calEOS_perfect

def subsonic_inlet(blk,face):

    p = blk.array['q'][:,:,:,0]
    p[fs[face]['s0_']] = 2.0*p[fs[face]['s1_']] - p[fs[face]['s2_']]

    #Update density
    EOS_ideal(blk,face,'PT')
    #Update momentum
    momentum(blk,face,'u')
    #Update total energy
    calEOS_perfect(blk,face,'PT')
