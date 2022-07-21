from .compute import utils


def RHS(mb):

    for blk in mb:
        # Zero out dQ array
        utils.dQzero(blk)

        # Primary advective fluxes
        mb.primaryAdvFlux(blk)
        # Apply strict advective boundary conditions
        for face in blk.faces:
            face.bcFunc(blk, face, mb.eos, mb.thtrdat, "strict", mb.tme)
        mb.applyPrimaryAdvFlux(blk, 1.0)  # <-- 1.0 is for primary flux

        # Secondary advective fluxes
        mb.secondaryAdvFlux(blk)
        mb.applySecondaryAdvFlux(blk, 0.0)  # <-- 0.0 is for secondary flux

        # Diffusive fluxes
        mb.diffFlux(blk)
        mb.applyDiffFlux(blk, -1.0)  # <-- -1.0 is arbitrary, see applyFlux.cpp

        # Chemical source terms
        mb.expChem(
            blk,
            mb.thtrdat,
            nChemSubSteps=mb.config["thermochem"]["nChemSubSteps"],
            dt=mb.config["simulation"]["dt"],
        )
