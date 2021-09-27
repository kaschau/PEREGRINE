from .compute import flux


def RHS(mb):

    # Zero out dQ array
    for blk in mb:
        flux.dQzero(blk)

    # Advective fluxes
    for blk in mb:
        flux.advective(blk, mb.thtrdat)

    # Diffusive fluxes
    if mb.config["RHS"]["diffusion"]:
        for blk in mb:
            flux.diffusive(blk, mb.thtrdat)

    # Chemical source terms
    if mb.config["thermochem"]["chemistry"]:
        for blk in mb:
            mb.chem(blk, mb.thtrdat)
