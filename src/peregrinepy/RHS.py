from .compute import utils


def RHS(mb):

    # Zero out dQ array
    for blk in mb:
        utils.dQzero(blk)

    # Primary advective fluxes
    for blk in mb:
        mb.primaryAdvFlux(blk, mb.thtrdat, 1.0)
    # Secondary advective fluxes
    for blk in mb:
        mb.secondaryAdvFlux(blk, mb.thtrdat, 0.0)

    # Diffusive fluxes
    for blk in mb:
        mb.diffFlux(blk, mb.thtrdat)

    # Chemical source terms
    for blk in mb:
        mb.expChem(blk, mb.thtrdat)
