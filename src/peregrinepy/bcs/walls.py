from .face_slice import fs

def adiabatic_noslip_wall(eos,blk,face,thermdat,terms):

    if terms == 'euler':
        p = blk.array['q'][:,:,:,0]
        p[fs[face]['s0_']] = p[fs[face]['s1_']]

        u = blk.array['q'][:,:,:,1:4]
        u[fs[face]['s0_']] = -u[fs[face]['s1_']]

        T = blk.array['q'][:,:,:,4]
        T[fs[face]['s0_']] = T[fs[face]['s1_']]

        #Update conservatives
        eos(blk,thermdat,face,'prims')

    elif terms == 'viscous':
        #extrapolate velocity gradient
        dvelodx = blk.array['dqdx'][:,:,:,1:4]
        dvelody = blk.array['dqdy'][:,:,:,1:4]
        dvelodz = blk.array['dqdz'][:,:,:,1:4]
        dvelodx[fs[face]['s0_']] = 2.0*dvelodx[fs[face]['s1_']] - dvelodx[fs[face]['s2_']]
        dvelody[fs[face]['s0_']] = 2.0*dvelody[fs[face]['s1_']] - dvelody[fs[face]['s2_']]
        dvelodz[fs[face]['s0_']] = 2.0*dvelodz[fs[face]['s1_']] - dvelodz[fs[face]['s2_']]
        #negate temp and species gradient (so gradient evaluates to zero on wall)
        dTNdx = blk.array['dqdx'][:,:,:,4::]
        dTNdy = blk.array['dqdy'][:,:,:,4::]
        dTNdz = blk.array['dqdz'][:,:,:,4::]
        dTNdx[fs[face]['s0_']] = - dTNdx[fs[face]['s1_']]
        dTNdy[fs[face]['s0_']] = - dTNdy[fs[face]['s1_']]
        dTNdz[fs[face]['s0_']] = - dTNdz[fs[face]['s1_']]

def adiabatic_slip_wall(eos,blk,face,thermdat,terms):

    if terms == 'euler':
        p = blk.array['q'][:,:,:,0]
        p[fs[face]['s0_']] = p[fs[face]['s1_']]

        u = blk.array['q'][:,:,:,1]
        v = blk.array['q'][:,:,:,2]
        w = blk.array['q'][:,:,:,3]
        if face in ['1','2']:
            nx = blk.array['inx']
            ny = blk.array['iny']
            nz = blk.array['inz']
        elif face in ['3','4']:
            nx = blk.array['jnx']
            ny = blk.array['jny']
            nz = blk.array['jnz']
        elif face in ['5','6']:
            nx = blk.array['knx']
            ny = blk.array['kny']
            nz = blk.array['knz']
        else:
            raise ValueError('Unknown face')
        u[fs[face]['s0_']] = u[fs[face]['s1_']] - 2.0*u[fs[face]['s1_']] * nx[fs[face]['s1_']]
        v[fs[face]['s0_']] = v[fs[face]['s1_']] - 2.0*v[fs[face]['s1_']] * ny[fs[face]['s1_']]
        w[fs[face]['s0_']] = w[fs[face]['s1_']] - 2.0*w[fs[face]['s1_']] * nz[fs[face]['s1_']]

        T = blk.array['q'][:,:,:,4]
        T[fs[face]['s0_']] = T[fs[face]['s1_']]

        #Update conservatives
        eos(blk,thermdat,face,'prims')

    elif terms == 'viscous':
        #negate velocity gradient (so gradient evaluates to zero on wall)
        dvelodx = blk.array['dqdx'][:,:,:,1:4]
        dvelody = blk.array['dqdy'][:,:,:,1:4]
        dvelodz = blk.array['dqdz'][:,:,:,1:4]
        dvelodx[fs[face]['s0_']] = - dvelodx[fs[face]['s1_']]
        dvelody[fs[face]['s0_']] = - dvelody[fs[face]['s1_']]
        dvelodz[fs[face]['s0_']] = - dvelodz[fs[face]['s1_']]
        #negate temp and species gradient (so gradient evaluates to zero on wall)
        dTNdx = blk.array['dqdx'][:,:,:,4::]
        dTNdy = blk.array['dqdy'][:,:,:,4::]
        dTNdz = blk.array['dqdz'][:,:,:,4::]
        dTNdx[fs[face]['s0_']] = - dTNdx[fs[face]['s1_']]
        dTNdy[fs[face]['s0_']] = - dTNdy[fs[face]['s1_']]
        dTNdz[fs[face]['s0_']] = - dTNdz[fs[face]['s1_']]

def adiabatic_moving_wall(eos,blk,face,thermdat,terms):

    if terms == 'euler':
        p = blk.array['q'][:,:,:,0]
        p[fs[face]['s0_']] = p[fs[face]['s1_']]

        u = blk.array['q'][:,:,:,1]
        v = blk.array['q'][:,:,:,2]
        w = blk.array['q'][:,:,:,3]
        if face in ['1','2']:
            nx = blk.array['inx']
            ny = blk.array['iny']
            nz = blk.array['inz']
        elif face in ['3','4']:
            nx = blk.array['jnx']
            ny = blk.array['jny']
            nz = blk.array['jnz']
        elif face in ['5','6']:
            nx = blk.array['knx']
            ny = blk.array['kny']
            nz = blk.array['knz']
        else:
            raise ValueError('Unknown face')
        u[fs[face]['s0_']] = 5.0
        v[fs[face]['s0_']] = v[fs[face]['s1_']] - 2.0*v[fs[face]['s1_']] * ny[fs[face]['s1_']]
        w[fs[face]['s0_']] = w[fs[face]['s1_']] - 2.0*w[fs[face]['s1_']] * nz[fs[face]['s1_']]

        T = blk.array['q'][:,:,:,4]
        T[fs[face]['s0_']] = T[fs[face]['s1_']]

        #Update conservatives
        eos(blk,thermdat,face,'prims')

    elif terms == 'viscous':
        #extrapolate velocity gradient
        dvelodx = blk.array['dqdx'][:,:,:,1:4]
        dvelody = blk.array['dqdy'][:,:,:,1:4]
        dvelodz = blk.array['dqdz'][:,:,:,1:4]
        dvelodx[fs[face]['s0_']] = 2.0*dvelodx[fs[face]['s1_']] - dvelodx[fs[face]['s2_']]
        dvelody[fs[face]['s0_']] = 2.0*dvelody[fs[face]['s1_']] - dvelody[fs[face]['s2_']]
        dvelodz[fs[face]['s0_']] = 2.0*dvelodz[fs[face]['s1_']] - dvelodz[fs[face]['s2_']]
        #negate temp and species gradient (so gradient evaluates to zero on wall)
        dTNdx = blk.array['dqdx'][:,:,:,4::]
        dTNdy = blk.array['dqdy'][:,:,:,4::]
        dTNdz = blk.array['dqdz'][:,:,:,4::]
        dTNdx[fs[face]['s0_']] = - dTNdx[fs[face]['s1_']]
        dTNdy[fs[face]['s0_']] = - dTNdy[fs[face]['s1_']]
        dTNdz[fs[face]['s0_']] = - dTNdz[fs[face]['s1_']]

def isoT_moving_wall(eos,blk,face,thermdat,terms):

    if terms == 'euler':
        p = blk.array['q'][:,:,:,0]
        p[fs[face]['s0_']] = p[fs[face]['s1_']]

        u = blk.array['q'][:,:,:,1]
        v = blk.array['q'][:,:,:,2]
        w = blk.array['q'][:,:,:,3]
        if face in ['1','2']:
            nx = blk.array['inx']
            ny = blk.array['iny']
            nz = blk.array['inz']
        elif face in ['3','4']:
            nx = blk.array['jnx']
            ny = blk.array['jny']
            nz = blk.array['jnz']
        elif face in ['5','6']:
            nx = blk.array['knx']
            ny = blk.array['kny']
            nz = blk.array['knz']
        else:
            raise ValueError('Unknown face')
        u[fs[face]['s0_']] = 5.0
        v[fs[face]['s0_']] = v[fs[face]['s1_']] - 2.0*v[fs[face]['s1_']] * ny[fs[face]['s1_']]
        w[fs[face]['s0_']] = w[fs[face]['s1_']] - 2.0*w[fs[face]['s1_']] * nz[fs[face]['s1_']]

        T = blk.array['q'][:,:,:,4]
        T[fs[face]['s0_']] = 1000.0

        #Update conservatives
        eos(blk,thermdat,face,'prims')

    elif terms == 'viscous':
        #extrapolate velocity gradient
        dvelodx = blk.array['dqdx'][:,:,:,1:4]
        dvelody = blk.array['dqdy'][:,:,:,1:4]
        dvelodz = blk.array['dqdz'][:,:,:,1:4]
        dvelodx[fs[face]['s0_']] = 2.0*dvelodx[fs[face]['s1_']] - dvelodx[fs[face]['s2_']]
        dvelody[fs[face]['s0_']] = 2.0*dvelody[fs[face]['s1_']] - dvelody[fs[face]['s2_']]
        dvelodz[fs[face]['s0_']] = 2.0*dvelodz[fs[face]['s1_']] - dvelodz[fs[face]['s2_']]
        #extrapolate temp and species gradient
        dTNdx = blk.array['dqdx'][:,:,:,4::]
        dTNdy = blk.array['dqdy'][:,:,:,4::]
        dTNdz = blk.array['dqdz'][:,:,:,4::]
        dTNdx[fs[face]['s0_']] = 2.0*dTNdx[fs[face]['s1_']] - dTNdx[fs[face]['s2_']]
        dTNdy[fs[face]['s0_']] = 2.0*dTNdy[fs[face]['s1_']] - dTNdy[fs[face]['s2_']]
        dTNdz[fs[face]['s0_']] = 2.0*dTNdz[fs[face]['s1_']] - dTNdz[fs[face]['s2_']]
