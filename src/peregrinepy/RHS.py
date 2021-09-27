from .compute import utils


def RHS(mb, nochem=False):

    # Zero out dQ array
    for blk in mb:
        utils.dQzero(blk)

    # Non dissipative advective fluxes
    for blk in mb:
        mb.nonDissAdvFlux(blk, mb.thtrdat)
    # Dissipative advective fluxes
    for blk in mb:
        mb.DissAdvFlux(blk, mb.thtrdat)

    # Diffusive fluxes
    for blk in mb:
        mb.diffFlux(blk, mb.thtrdat)

    # Chemical source terms
    if not nochem:
        for blk in mb:
            mb.chem(blk, mb.thtrdat)
