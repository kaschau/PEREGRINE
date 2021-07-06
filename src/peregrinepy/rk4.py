from .RHS import RHS
from .consistify import consistify


def step(mb,dt,config):

    #store zeroth stage solution
    for blk in mb:
        blk.array['rhs0'][:] = blk.array['Q'][:]


    # First Stage
    for blk in mb:
        blk.array['dQ'][:,:,:,:] = 0.0

    RHS(mb,config)

    for blk in mb:
        blk.array['rhs1'][:] = dt*blk.array['dQ']
        blk.array['Q'][:] = blk.array['rhs0'] + 0.5*blk.array['rhs1']
    consistify(mb,config)

    # Second Stage
    for blk in mb:
        blk.array['dQ'][:,:,:,:] = 0.0

    RHS(mb,config)

    for blk in mb:
        blk.array['rhs2'][:] = dt*blk.array['dQ']
        blk.array['Q'][:] = blk.array['rhs0'] + 0.5*blk.array['rhs2']
    consistify(mb,config)

    # Third Stage
    for blk in mb:
        blk.array['dQ'][:,:,:,:] = 0.0

    RHS(mb,config)

    for blk in mb:
        blk.array['rhs3'][:] = dt*blk.array['dQ']
        blk.array['Q'][:] = blk.array['rhs0'] + blk.array['rhs3']
    consistify(mb,config)

    # Fourth Stage
    for blk in mb:
        blk.array['dQ'][:,:,:,:] = 0.0

    RHS(mb,config)

    for blk in mb:
        blk.array['Q'][:] = blk.array['rhs0'] + (     blk.array['rhs1'] +
                                                  2.0*blk.array['rhs2'] +
                                                  2.0*blk.array['rhs3'] +
                                                  dt *blk.array['dQ']   )/6.0
    consistify(mb,config)

    mb.nrt += 1
    mb.tme += dt
