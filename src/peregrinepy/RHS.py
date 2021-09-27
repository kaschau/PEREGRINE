from .compute import utils


def RHS(mb):

    # Zero out dQ array
    for blk in mb:
        utils.dQzero(blk)

    # Non dissipative advective fluxes
    for blk in mb:
        mb.nonDissAdvFlx(blk, mb.thtrdat)
    # Dissipative advective fluxes
    for blk in mb:
        mb.DissAdvFlx(blk, mb.thtrdat)

    # Diffusive fluxes
    for blk in mb:
        mb.diffFlx(blk, mb.thtrdat)

    # Chemical source terms
    for blk in mb:
        mb.chem(blk, mb.thtrdat)
