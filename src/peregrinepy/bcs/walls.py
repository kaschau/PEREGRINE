

def adiabatic_noslip_wall(eos,blk,face,thtrdat,terms):

    nface = face.nface

    if terms == 'euler':
        p = blk.array['q'][:,:,:,0]
        p[face.s0_] = p[face.s1_]

        u = blk.array['q'][:,:,:,1:4]
        u[face.s0_] = -u[face.s1_]

        TN = blk.array['q'][:,:,:,4::]
        TN[face.s0_] = TN[face.s1_]

        #Update conservatives
        eos(blk,thtrdat,nface,'prims')

    elif terms == 'viscous':
        #extrapolate velocity gradient
        dvelodx = blk.array['dqdx'][:,:,:,1:4]
        dvelody = blk.array['dqdy'][:,:,:,1:4]
        dvelodz = blk.array['dqdz'][:,:,:,1:4]
        dvelodx[face.s0_] = 2.0*dvelodx[face.s1_] - dvelodx[face.s2_]
        dvelody[face.s0_] = 2.0*dvelody[face.s1_] - dvelody[face.s2_]
        dvelodz[face.s0_] = 2.0*dvelodz[face.s1_] - dvelodz[face.s2_]
        #negate temp and species gradient (so gradient evaluates to zero on wall)
        dTNdx = blk.array['dqdx'][:,:,:,4::]
        dTNdy = blk.array['dqdy'][:,:,:,4::]
        dTNdz = blk.array['dqdz'][:,:,:,4::]
        dTNdx[face.s0_] = - dTNdx[face.s1_]
        dTNdy[face.s0_] = - dTNdy[face.s1_]
        dTNdz[face.s0_] = - dTNdz[face.s1_]

def adiabatic_slip_wall(eos,blk,face,thtrdat,terms):

    nface = face.nface

    if terms == 'euler':
        p = blk.array['q'][:,:,:,0]
        p[face.s0_] = p[face.s1_]

        u = blk.array['q'][:,:,:,1]
        v = blk.array['q'][:,:,:,2]
        w = blk.array['q'][:,:,:,3]
        if nface in [1,2]:
            nx = blk.array['inx']
            ny = blk.array['iny']
            nz = blk.array['inz']
        elif nface in [3,4]:
            nx = blk.array['jnx']
            ny = blk.array['jny']
            nz = blk.array['jnz']
        elif nface in [5,6]:
            nx = blk.array['knx']
            ny = blk.array['kny']
            nz = blk.array['knz']
        else:
            raise ValueError('Unknown nface')
        u[face.s0_] = u[face.s1_] - 2.0*u[face.s1_] * nx[face.s1_]
        v[face.s0_] = v[face.s1_] - 2.0*v[face.s1_] * ny[face.s1_]
        w[face.s0_] = w[face.s1_] - 2.0*w[face.s1_] * nz[face.s1_]

        TN = blk.array['q'][:,:,:,4::]
        TN[face.s0_] = TN[face.s1_]

        #Update conservatives
        eos(blk,thtrdat,nface,'prims')

    elif terms == 'viscous':
        #negate velocity gradient (so gradient evaluates to zero on wall)
        dvelodx = blk.array['dqdx'][:,:,:,1:4]
        dvelody = blk.array['dqdy'][:,:,:,1:4]
        dvelodz = blk.array['dqdz'][:,:,:,1:4]
        dvelodx[face.s0_] = - dvelodx[face.s1_]
        dvelody[face.s0_] = - dvelody[face.s1_]
        dvelodz[face.s0_] = - dvelodz[face.s1_]
        #negate temp and species gradient (so gradient evaluates to zero on wall)
        dTNdx = blk.array['dqdx'][:,:,:,4::]
        dTNdy = blk.array['dqdy'][:,:,:,4::]
        dTNdz = blk.array['dqdz'][:,:,:,4::]
        dTNdx[face.s0_] = - dTNdx[face.s1_]
        dTNdy[face.s0_] = - dTNdy[face.s1_]
        dTNdz[face.s0_] = - dTNdz[face.s1_]

def adiabatic_moving_wall(eos, blk, face, thtrdat, terms):

    nface = face.nface

    if terms == 'euler':
        p = blk.array['q'][:,:,:,0]
        p[face.s0_] = p[face.s1_]

        u = blk.array['q'][:,:,:,1]
        v = blk.array['q'][:,:,:,2]
        w = blk.array['q'][:,:,:,3]
        if nface in [1,2]:
            nx = blk.array['inx']
            ny = blk.array['iny']
            nz = blk.array['inz']
        elif nface in [3,4]:
            nx = blk.array['jnx']
            ny = blk.array['jny']
            nz = blk.array['jnz']
        elif nface in [5,6]:
            nx = blk.array['knx']
            ny = blk.array['kny']
            nz = blk.array['knz']
        else:
            raise ValueError('Unknown nface')
        u[face.s0_] = 2.0*face.bcvals['u'] - u[face.s1_]
        v[face.s0_] = 2.0*face.bcvals['v'] - v[face.s1_]
        w[face.s0_] = 2.0*face.bcvals['w'] - w[face.s1_]

        TN = blk.array['q'][:,:,:,4::]
        TN[face.s0_] = TN[face.s1_]

        #Update conservatives
        eos(blk, thtrdat, nface, 'prims')

    elif terms == 'viscous':
        #extrapolate velocity gradient
        dvelodx = blk.array['dqdx'][:,:,:,1:4]
        dvelody = blk.array['dqdy'][:,:,:,1:4]
        dvelodz = blk.array['dqdz'][:,:,:,1:4]
        dvelodx[face.s0_] = 2.0*dvelodx[face.s1_] - dvelodx[face.s2_]
        dvelody[face.s0_] = 2.0*dvelody[face.s1_] - dvelody[face.s2_]
        dvelodz[face.s0_] = 2.0*dvelodz[face.s1_] - dvelodz[face.s2_]

        #negate temp and species gradient (so gradient evaluates to zero on wall)
        dTNdx = blk.array['dqdx'][:,:,:,4::]
        dTNdy = blk.array['dqdy'][:,:,:,4::]
        dTNdz = blk.array['dqdz'][:,:,:,4::]
        dTNdx[face.s0_] = - dTNdx[face.s1_]
        dTNdy[face.s0_] = - dTNdy[face.s1_]
        dTNdz[face.s0_] = - dTNdz[face.s1_]

def isoT_moving_wall(eos,blk,face,thtrdat,terms):

    nface = face.nface

    if terms == 'euler':
        p = blk.array['q'][:,:,:,0]
        p[face.s0_] = p[face.s1_]

        u = blk.array['q'][:,:,:,1]
        v = blk.array['q'][:,:,:,2]
        w = blk.array['q'][:,:,:,3]
        if nface in [1,2]:
            nx = blk.array['inx']
            ny = blk.array['iny']
            nz = blk.array['inz']
        elif nface in [3,4]:
            nx = blk.array['jnx']
            ny = blk.array['jny']
            nz = blk.array['jnz']
        elif nface in [5,6]:
            nx = blk.array['knx']
            ny = blk.array['kny']
            nz = blk.array['knz']
        else:
            raise ValueError('Unknown nface')
        u[face.s0_] = 2.0*face.bcvals['u'] - u[face.s1_]
        v[face.s0_] = 2.0*face.bcvals['v'] - v[face.s1_]
        w[face.s0_] = 2.0*face.bcvals['w'] - w[face.s1_]

        #Match species
        N = blk.array['q'][:,:,:,4::]
        N[face.s0_] =  N[face.s1_]

        #Set tempeerature
        T = blk.array['q'][:,:,:,4]
        T[face.s0_] = 2.0*face.bcvals['T'] - T[face.s1_]

        #Update conservatives
        eos(blk,thtrdat,nface,'prims')

    elif terms == 'viscous':
        #extrapolate velocity gradient
        dvelodx = blk.array['dqdx'][:,:,:,1:4]
        dvelody = blk.array['dqdy'][:,:,:,1:4]
        dvelodz = blk.array['dqdz'][:,:,:,1:4]
        dvelodx[face.s0_] = 2.0*dvelodx[face.s1_] - dvelodx[face.s2_]
        dvelody[face.s0_] = 2.0*dvelody[face.s1_] - dvelody[face.s2_]
        dvelodz[face.s0_] = 2.0*dvelodz[face.s1_] - dvelodz[face.s2_]
        #extrapolate temp and species gradient
        dTNdx = blk.array['dqdx'][:,:,:,4::]
        dTNdy = blk.array['dqdy'][:,:,:,4::]
        dTNdz = blk.array['dqdz'][:,:,:,4::]
        dTNdx[face.s0_] = 2.0*dTNdx[face.s1_] - dTNdx[face.s2_]
        dTNdy[face.s0_] = 2.0*dTNdy[face.s1_] - dTNdy[face.s2_]
        dTNdz[face.s0_] = 2.0*dTNdz[face.s1_] - dTNdz[face.s2_]
