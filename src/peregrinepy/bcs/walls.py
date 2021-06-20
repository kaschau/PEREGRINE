from .face_slice import fs
from ..compute_ import momentum, EOS_ideal, calEOS_perfect


def adiabatic_noslip_wall(blk,face):

    p = blk.array['q'][:,:,:,0]
    p[fs[face]['s0_']] = p[fs[face]['s1_']]

    u = blk.array['q'][:,:,:,1:4]
    u[fs[face]['s0_']] = -u[fs[face]['s1_']]

    T = blk.array['q'][:,:,:,4]
    T[fs[face]['s0_']] = T[fs[face]['s1_']]

    #Update density
    EOS_ideal(blk,face,'PT')
    #Update momentum
    momentum(blk,face,'u')
    #Update total energy
    calEOS_perfect(blk,face,'T')
