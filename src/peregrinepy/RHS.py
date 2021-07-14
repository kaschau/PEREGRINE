from .compute import advective,apply_flux


def RHS(mb):

    advective(mb)
    apply_flux(mb)
