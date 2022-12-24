from .compute import utils


def RHS(mb):

    for blk in mb:
        # Zero out dQ array
        utils.dQzero(blk)

        # Primary advective fluxes
        mb.primaryAdvFlux(blk)
        mb.applyPrimaryAdvFlux(blk, 1.0)  # <-- 1.0 is for primary flux

        # Secondary advective fluxes
        mb.secondaryAdvFlux(blk)
        mb.applySecondaryAdvFlux(blk, 0.0)  # <-- 0.0 is for secondary flux

        # Apply viscous boundary conditions
        for face in blk.faces:
            face.bcFunc(blk, face, mb.eos, mb.thtrdat, "preDqDxyz", mb.tme)

        # Update spatial derivatives
        mb.dqdxyz(blk)
        communicate(mb, mb.dqdxyzCommList)
        # Apply spatial derivative boundary conditions
        for face in blk.faces:
            face.bcFunc(blk, face, mb.eos, mb.thtrdat, "postDqDxyz", mb.tme)

        # Diffusive fluxes
        mb.diffFlux(blk)
        mb.applyDiffFlux(blk, -1.0)  # <-- -1.0 is arbitrary, see applyFlux.cpp

        # Chemical source terms
        mb.expChem(
            blk,
            mb.thtrdat,
            nChemSubSteps=mb.config["thermochem"]["nChemSubSteps"],
            dt=mb.config["timeIntegration"]["dt"],
        )
