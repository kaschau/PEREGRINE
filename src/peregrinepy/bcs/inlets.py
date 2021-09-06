

def constant_velocity_subsonic_inlet(eos,blk,face,thtrdat,terms):

    nface = face.nface

    if terms == 'euler':
        #extrapolate pressure
        p = blk.array['q'][:,:,:,0]
        p[face.s0_] = 2.0*p[face.s1_] - p[face.s2_]

        #apply velo on face
        u = blk.array['q'][:,:,:,1]
        v = blk.array['q'][:,:,:,2]
        w = blk.array['q'][:,:,:,3]
        u[face.s0_] = 2.0*face.bcvals['u'] - u[face.s1_]
        v[face.s0_] = 2.0*face.bcvals['v'] - v[face.s1_]
        w[face.s0_] = 2.0*face.bcvals['w'] - w[face.s1_]

        T = blk.array['q'][:,:,:,4]
        T[face.s0_] = 2.0*face.bcvals['T'] - T[face.s1_]

        for sn,n in enumerate(thtrdat.species_names[0:-1]):
            N = blk.array['q'][:,:,:,5+n]
            N[face.s0_] = 2.0*face.bcvals[sn[n]] - N[face.s1_]

        #Update conserved
        eos(blk,thtrdat,nface,'prims')

    elif terms == 'viscous':
        #neumann all gradients
        blk.array['dqdx'][face.s0_] = blk.array['dqdx'][face.s1_]
        blk.array['dqdy'][face.s0_] = blk.array['dqdy'][face.s1_]
        blk.array['dqdz'][face.s0_] = blk.array['dqdz'][face.s1_]
