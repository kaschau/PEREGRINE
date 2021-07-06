from .RHS import RHS
from .consistify import consistify


def step(mb,dt,config):

    #Zero it out dQ
    for blk in mb:
        blk.array['dQ'][:,:,:,:] = 0.0

    RHS(mb,config)
    #rhs is computed, it is stored in dQ

    #add it to current solution
    for blk in mb:
        blk.array['Q'][:] += dt*blk.array['dQ']

    mb.nrt += 1
    mb.tme += dt

    consistify(mb,config)
