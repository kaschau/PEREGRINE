from .RHS import RHS
from .consistify import consistify


def step(mb,dt,config):

    #Point dQ to the first rk stage and zero it out
    for blk in mb:
        blk.dQ = blk.rhs0
        blk.array['dQ'] = blk.array['rhs0']
        blk.array['dQ'][:,:,:,:] = 0.0

    RHS(mb,config)
    #rhs is computed, it is stored in dQ (or rhs0).

    #add it to current solution
    for blk in mb:
        blk.array['Q'] += dt*blk.array['dQ']

    consistify(mb,config)
