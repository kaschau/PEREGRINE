from .compute import flux


def RHS(mb):

    # Zero out dQ array
    flux.dQzero(mb)

    flux.advective(mb, mb.thtrdat)

    if mb.config["RHS"]["diffusion"]:
        flux.diffusive(mb, mb.thtrdat)

    if mb.config["thermochem"]["chemistry"]:
        mb.chem(mb, mb.thtrdat)
