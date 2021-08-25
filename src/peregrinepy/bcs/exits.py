from .face_slice import fs

def constant_pressure_subsonic_exit(eos,blk,face,thermdat,terms):

    nface = face.nface

    if terms == 'euler':
        #Extrapolate everything
        q = blk.array['q'][:,:,:,1::]
        q[fs[nface]['s0_']] = 2.0*q[fs[nface]['s1_']] - q[fs[nface]['s2_']]

        #Set pressure at face
        p = blk.array['q'][:,:,:,0]
        p[fs[nface]['s0_']] = 2.0*face.bc['values']['p'] - p[fs[nface]['s1_']]

        #Update conservatives
        eos(blk,thermdat,nface,'prims')

    elif terms == 'viscous':
        #neumann all gradients
        blk.array['dqdx'][fs[nface]['s0_']] = blk.array['dqdx'][fs[nface]['s1_']]
        blk.array['dqdy'][fs[nface]['s0_']] = blk.array['dqdy'][fs[nface]['s1_']]
        blk.array['dqdz'][fs[nface]['s0_']] = blk.array['dqdz'][fs[nface]['s1_']]
