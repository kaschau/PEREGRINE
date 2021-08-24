from .face_slice import fs

def constant_velocity_subsonic_inlet(eos,blk,face,thermdat,terms):

    nface = face.nface

    if terms == 'euler':
        p = blk.array['q'][:,:,:,0]
        p[fs[nface]['s0_']] = 2.0*p[fs[nface]['s1_']] - p[fs[nface]['s2_']]

        #Update conserved
        eos(blk,thermdat,nface,'prims')

    elif terms == 'viscous':
        #neumann all gradients
        blk.array['dqdx'][fs[nface]['s0_']] = blk.array['dqdx'][fs[nface]['s1_']]
        blk.array['dqdy'][fs[nface]['s0_']] = blk.array['dqdy'][fs[nface]['s1_']]
        blk.array['dqdz'][fs[nface]['s0_']] = blk.array['dqdz'][fs[nface]['s1_']]
