from .face_slice import fs

def constant_pressure_subsonic_exit(eos,blk,nface,thermdat,terms):

    if terms == 'euler':
        q = blk.array['q'][:,:,:,1::]
        q[fs[nface]['s0_']] = 2.0*q[fs[nface]['s1_']] - q[fs[nface]['s2_']]

        #Update conservatives
        eos(blk,thermdat,nface,'prims')

    elif terms == 'viscous':
        #neumann all gradients
        blk.array['dqdx'][fs[nface]['s0_']] = blk.array['dqdx'][fs[nface]['s1_']]
        blk.array['dqdy'][fs[nface]['s0_']] = blk.array['dqdy'][fs[nface]['s1_']]
        blk.array['dqdz'][fs[nface]['s0_']] = blk.array['dqdz'][fs[nface]['s1_']]
