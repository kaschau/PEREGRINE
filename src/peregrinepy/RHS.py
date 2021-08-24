from .compute import dQzero,advective,diffusive

def RHS(mb):

    #Zero out dQ array
    dQzero(mb)

    advective(mb,mb.thermdat)

    if mb.config['RHS']['diffusion']:
        diffusive(mb,mb.thermdat)
