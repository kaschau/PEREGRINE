from .face_slice import fs

def subsonic_exit(blk,face):

    q = blk.array['q'][:,:,:,1::]
    q[fs[face]['s0_']] = 2.0*q[fs[face]['s1_']] - q[fs[face]['s2_']]
