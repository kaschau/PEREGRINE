from .face_slice import fs

def constant_velocity_subsonic_inlet(eos,blk,face,thermdat,terms):

    nface = face.nface

    if terms == 'euler':
        #extrapolate pressure
        p = blk.array['q'][:,:,:,0]
        p[fs[nface]['s0_']] = 2.0*p[fs[nface]['s1_']] - p[fs[nface]['s2_']]

        #apply velo on face
        u = blk.array['q'][:,:,:,1]
        v = blk.array['q'][:,:,:,2]
        w = blk.array['q'][:,:,:,3]
        u[fs[nface]['s0_']] = 2.0*face.bc['values']['u'] - u[fs[nface]['s1_']]
        v[fs[nface]['s0_']] = 2.0*face.bc['values']['v'] - v[fs[nface]['s1_']]
        w[fs[nface]['s0_']] = 2.0*face.bc['values']['w'] - w[fs[nface]['s1_']]

        T = blk.array['q'][:,:,:,4]
        T[fs[nface]['s0_']] = 2.0*face.bc['values']['T'] - T[fs[nface]['s1_']]

        for sn,n in enumerate(thermdat.species_names[0:-1]):
            N = blk.array['q'][:,:,:,5+n]
            N[fs[nface]['s0_']] = 2.0*face.bc['values'][sn[n]] - N[fs[nface]['s1_']]

        #Update conserved
        eos(blk,thermdat,nface,'prims')

    elif terms == 'viscous':
        #neumann all gradients
        blk.array['dqdx'][fs[nface]['s0_']] = blk.array['dqdx'][fs[nface]['s1_']]
        blk.array['dqdy'][fs[nface]['s0_']] = blk.array['dqdy'][fs[nface]['s1_']]
        blk.array['dqdz'][fs[nface]['s0_']] = blk.array['dqdz'][fs[nface]['s1_']]
