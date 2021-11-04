from .compute import utils


def RHS(mb):

    # Zero out dQ array
    for blk in mb:
        utils.dQzero(blk)

        # Primary advective fluxes
        mb.primaryAdvFlux(blk, mb.thtrdat)
        mb.applyPrimaryAdvFlux(blk, primary=1.0)
        # Secondary advective fluxes
        mb.secondaryAdvFlux(blk, mb.thtrdat)
        mb.applySecondaryAdvFlux(blk, primary=0.0)

        # Diffusive fluxes
        mb.diffFlux(blk, mb.thtrdat)
        mb.applyDiffFlux(blk, primary=-1.0)

        # Chemical source terms
        mb.expChem(blk, mb.thtrdat)
