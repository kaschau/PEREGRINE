from .face_slice import fs

def subsonic_inlet(eos,blk,face,thermdat,terms):

    if terms == 'euler':
        p = blk.array['q'][:,:,:,0]
        p[fs[face]['s0_']] = 2.0*p[fs[face]['s1_']] - p[fs[face]['s2_']]

        #Update conserved
        eos(blk,thermdat,face,'prims')

    elif terms == 'viscous':
        #neumann all gradients
        blk.array['dqdx'][fs[face]['s0_']] = blk.array['dqdx'][fs[face]['s1_']]
        blk.array['dqdy'][fs[face]['s0_']] = blk.array['dqdy'][fs[face]['s1_']]
        blk.array['dqdz'][fs[face]['s0_']] = blk.array['dqdz'][fs[face]['s1_']]
