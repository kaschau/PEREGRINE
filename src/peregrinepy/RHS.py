from .compute import utils


def RHS(mb):

    # Zero out dQ array
    for blk in mb:
        utils.dQzero(blk)

    # Non dissipative advective fluxes
    for blk in mb:
        mb.nonDissAdvFlux(blk, mb.thtrdat)
    # Dissipative advective fluxes
    for blk in mb:
        mb.dissAdvFlux(blk, mb.thtrdat)

    # Diffusive fluxes
    for blk in mb:
        mb.diffFlux(blk, mb.thtrdat)

    # Chemical source terms
    for blk in mb:
        mb.expchem(blk, mb.thtrdat)
