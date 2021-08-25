from .face_slice import fs

def adiabatic_noslip_wall(eos,blk,face,thermdat,terms):

    nface = face.nface

    if terms == 'euler':
        p = blk.array['q'][:,:,:,0]
        p[fs[nface]['s0_']] = p[fs[nface]['s1_']]

        u = blk.array['q'][:,:,:,1:4]
        u[fs[nface]['s0_']] = -u[fs[nface]['s1_']]

        TN = blk.array['q'][:,:,:,4::]
        TN[fs[nface]['s0_']] = TN[fs[nface]['s1_']]

        #Update conservatives
        eos(blk,thermdat,nface,'prims')

    elif terms == 'viscous':
        #extrapolate velocity gradient
        dvelodx = blk.array['dqdx'][:,:,:,1:4]
        dvelody = blk.array['dqdy'][:,:,:,1:4]
        dvelodz = blk.array['dqdz'][:,:,:,1:4]
        dvelodx[fs[nface]['s0_']] = 2.0*dvelodx[fs[nface]['s1_']] - dvelodx[fs[nface]['s2_']]
        dvelody[fs[nface]['s0_']] = 2.0*dvelody[fs[nface]['s1_']] - dvelody[fs[nface]['s2_']]
        dvelodz[fs[nface]['s0_']] = 2.0*dvelodz[fs[nface]['s1_']] - dvelodz[fs[nface]['s2_']]
        #negate temp and species gradient (so gradient evaluates to zero on wall)
        dTNdx = blk.array['dqdx'][:,:,:,4::]
        dTNdy = blk.array['dqdy'][:,:,:,4::]
        dTNdz = blk.array['dqdz'][:,:,:,4::]
        dTNdx[fs[nface]['s0_']] = - dTNdx[fs[nface]['s1_']]
        dTNdy[fs[nface]['s0_']] = - dTNdy[fs[nface]['s1_']]
        dTNdz[fs[nface]['s0_']] = - dTNdz[fs[nface]['s1_']]

def adiabatic_slip_wall(eos,blk,face,thermdat,terms):

    nface = face.nface

    if terms == 'euler':
        p = blk.array['q'][:,:,:,0]
        p[fs[nface]['s0_']] = p[fs[nface]['s1_']]

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
        u[fs[nface]['s0_']] = u[fs[nface]['s1_']] - 2.0*u[fs[nface]['s1_']] * nx[fs[nface]['s1_']]
        v[fs[nface]['s0_']] = v[fs[nface]['s1_']] - 2.0*v[fs[nface]['s1_']] * ny[fs[nface]['s1_']]
        w[fs[nface]['s0_']] = w[fs[nface]['s1_']] - 2.0*w[fs[nface]['s1_']] * nz[fs[nface]['s1_']]

        TN = blk.array['q'][:,:,:,4::]
        TN[fs[nface]['s0_']] = TN[fs[nface]['s1_']]

        #Update conservatives
        eos(blk,thermdat,nface,'prims')

    elif terms == 'viscous':
        #negate velocity gradient (so gradient evaluates to zero on wall)
        dvelodx = blk.array['dqdx'][:,:,:,1:4]
        dvelody = blk.array['dqdy'][:,:,:,1:4]
        dvelodz = blk.array['dqdz'][:,:,:,1:4]
        dvelodx[fs[nface]['s0_']] = - dvelodx[fs[nface]['s1_']]
        dvelody[fs[nface]['s0_']] = - dvelody[fs[nface]['s1_']]
        dvelodz[fs[nface]['s0_']] = - dvelodz[fs[nface]['s1_']]
        #negate temp and species gradient (so gradient evaluates to zero on wall)
        dTNdx = blk.array['dqdx'][:,:,:,4::]
        dTNdy = blk.array['dqdy'][:,:,:,4::]
        dTNdz = blk.array['dqdz'][:,:,:,4::]
        dTNdx[fs[nface]['s0_']] = - dTNdx[fs[nface]['s1_']]
        dTNdy[fs[nface]['s0_']] = - dTNdy[fs[nface]['s1_']]
        dTNdz[fs[nface]['s0_']] = - dTNdz[fs[nface]['s1_']]

def adiabatic_moving_wall(eos, blk, face, thermdat, terms):

    nface = face.nface

    if terms == 'euler':
        p = blk.array['q'][:,:,:,0]
        p[fs[nface]['s0_']] = p[fs[nface]['s1_']]

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
        u[fs[nface]['s0_']] = 2.0*face.bc['values']['u'] - u[fs[nface]['s1_']]
        v[fs[nface]['s0_']] = 2.0*face.bc['values']['v'] - v[fs[nface]['s1_']]
        w[fs[nface]['s0_']] = 2.0*face.bc['values']['w'] - w[fs[nface]['s1_']]

        TN = blk.array['q'][:,:,:,4::]
        TN[fs[nface]['s0_']] = TN[fs[nface]['s1_']]

        #Update conservatives
        eos(blk, thermdat, nface, 'prims')

    elif terms == 'viscous':
        #extrapolate velocity gradient
        dvelodx = blk.array['dqdx'][:,:,:,1:4]
        dvelody = blk.array['dqdy'][:,:,:,1:4]
        dvelodz = blk.array['dqdz'][:,:,:,1:4]
        dvelodx[fs[nface]['s0_']] = 2.0*dvelodx[fs[nface]['s1_']] - dvelodx[fs[nface]['s2_']]
        dvelody[fs[nface]['s0_']] = 2.0*dvelody[fs[nface]['s1_']] - dvelody[fs[nface]['s2_']]
        dvelodz[fs[nface]['s0_']] = 2.0*dvelodz[fs[nface]['s1_']] - dvelodz[fs[nface]['s2_']]

        #negate temp and species gradient (so gradient evaluates to zero on wall)
        dTNdx = blk.array['dqdx'][:,:,:,4::]
        dTNdy = blk.array['dqdy'][:,:,:,4::]
        dTNdz = blk.array['dqdz'][:,:,:,4::]
        dTNdx[fs[nface]['s0_']] = - dTNdx[fs[nface]['s1_']]
        dTNdy[fs[nface]['s0_']] = - dTNdy[fs[nface]['s1_']]
        dTNdz[fs[nface]['s0_']] = - dTNdz[fs[nface]['s1_']]

def isoT_moving_wall(eos,blk,face,thermdat,terms):

    nface = face.nface

    if terms == 'euler':
        p = blk.array['q'][:,:,:,0]
        p[fs[nface]['s0_']] = p[fs[nface]['s1_']]

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
        u[fs[nface]['s0_']] = 2.0*face.bc['values']['u'] - u[fs[nface]['s1_']]
        v[fs[nface]['s0_']] = 2.0*face.bc['values']['v'] - v[fs[nface]['s1_']]
        w[fs[nface]['s0_']] = 2.0*face.bc['values']['w'] - w[fs[nface]['s1_']]

        #Match species
        N = blk.array['q'][:,:,:,4::]
        N[fs[nface]['s0_']] =  N[fs[nface]['s1_']]

        #Set tempeerature
        T = blk.array['q'][:,:,:,4]
        T[fs[nface]['s0_']] = 2.0*face.bc['values']['T'] - T[fs[nface]['s1_']]

        #Update conservatives
        eos(blk,thermdat,nface,'prims')

    elif terms == 'viscous':
        #extrapolate velocity gradient
        dvelodx = blk.array['dqdx'][:,:,:,1:4]
        dvelody = blk.array['dqdy'][:,:,:,1:4]
        dvelodz = blk.array['dqdz'][:,:,:,1:4]
        dvelodx[fs[nface]['s0_']] = 2.0*dvelodx[fs[nface]['s1_']] - dvelodx[fs[nface]['s2_']]
        dvelody[fs[nface]['s0_']] = 2.0*dvelody[fs[nface]['s1_']] - dvelody[fs[nface]['s2_']]
        dvelodz[fs[nface]['s0_']] = 2.0*dvelodz[fs[nface]['s1_']] - dvelodz[fs[nface]['s2_']]
        #extrapolate temp and species gradient
        dTNdx = blk.array['dqdx'][:,:,:,4::]
        dTNdy = blk.array['dqdy'][:,:,:,4::]
        dTNdz = blk.array['dqdz'][:,:,:,4::]
        dTNdx[fs[nface]['s0_']] = 2.0*dTNdx[fs[nface]['s1_']] - dTNdx[fs[nface]['s2_']]
        dTNdy[fs[nface]['s0_']] = 2.0*dTNdy[fs[nface]['s1_']] - dTNdy[fs[nface]['s2_']]
        dTNdz[fs[nface]['s0_']] = 2.0*dTNdz[fs[nface]['s1_']] - dTNdz[fs[nface]['s2_']]
