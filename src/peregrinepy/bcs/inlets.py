from .face_slice import fs

def subsonic_inlet(eos,blk,face,thermdat):

    p = blk.array['q'][:,:,:,0]
    p[fs[face]['s0_']] = 2.0*p[fs[face]['s1_']] - p[fs[face]['s2_']]

    #Update conserved
    eos(blk,thermdat,face,'prims')
