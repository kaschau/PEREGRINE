from .compute import advective,diffusive

def RHS(mb):

    advective(mb,mb.thermdat)

    diffusive(mb,mb.thermdat)
