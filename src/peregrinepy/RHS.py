from .compute import flux


def RHS(mb):

    # Zero out dQ array
    flux.dQzero(mb)

    flux.advective(mb, mb.thtrdat)

    if mb.config["RHS"]["diffusion"]:
        flux.diffusive(mb, mb.thtrdat)

    if mb.config["thermochem"]["chemistry"]:
        for blk in mb:
            mb.chem(blk, mb.thtrdat)
