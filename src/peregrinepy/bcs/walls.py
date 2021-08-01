from .face_slice import fs
from ..compute import cpg


def adiabatic_noslip_wall(blk,face,thermdat):

    p = blk.array['q'][:,:,:,0]
    p[fs[face]['s0_']] = p[fs[face]['s1_']]

    u = blk.array['q'][:,:,:,1:4]
    u[fs[face]['s0_']] = -u[fs[face]['s1_']]

    T = blk.array['q'][:,:,:,4]
    T[fs[face]['s0_']] = T[fs[face]['s1_']]

    #Update conservatives
    cpg(blk,thermdat,face,'prims')

def adiabatic_slip_wall(blk,face,thermdat):

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
    cpg(blk,thermdat,face,'prims')
