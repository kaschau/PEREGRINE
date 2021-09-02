from .compute import dQzero,advective,diffusive

def RHS(mb):

    #Zero out dQ array
    dQzero(mb)

    advective(mb,mb.thtrdat)

    if mb.config['RHS']['diffusion']:
        diffusive(mb,mb.thtrdat)
