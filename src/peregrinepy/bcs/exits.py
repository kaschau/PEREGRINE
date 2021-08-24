from .face_slice import fs

def subsonic_exit(eos,blk,face,thermdat,terms):

    if terms == 'euler':
        q = blk.array['q'][:,:,:,1::]
        q[fs[face]['s0_']] = 2.0*q[fs[face]['s1_']] - q[fs[face]['s2_']]

        #Update conservatives
        eos(blk,thermdat,face,'prims')

    elif terms == 'viscous':
        #neumann all gradients
        blk.array['dqdx'][fs[face]['s0_']] = blk.array['dqdx'][fs[face]['s1_']]
        blk.array['dqdy'][fs[face]['s0_']] = blk.array['dqdy'][fs[face]['s1_']]
        blk.array['dqdz'][fs[face]['s0_']] = blk.array['dqdz'][fs[face]['s1_']]
