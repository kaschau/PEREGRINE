from .compute import advective,apply_flux


def RHS(mb,config):

    advective(mb)
    apply_flux(mb)
