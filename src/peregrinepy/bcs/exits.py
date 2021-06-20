from .face_slice import fs
from ..compute_ import momentum, EOS_ideal, calEOS_perfect

def subsonic_exit(blk,face):

    q = blk.array['q'][:,:,:,1::]
    q[fs[face]['s0_']] = 2.0*q[fs[face]['s1_']] - q[fs[face]['s2_']]

    #Update density
    EOS_ideal(blk,face,'PT')
    #Update momentum
    momentum(blk,face,'u')
    #Update total energy
    calEOS_perfect(blk,face,'PT')
