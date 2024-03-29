from .compute import utils
from .mpiComm import communicate

"""
The right hand side of a peregrine case. Starts from dQ/dt=0 and
progressively flux differences based on the RHS attribute methods
set for the multiBlock solver.
"""


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

    if mb.config["RHS"]["diffusion"]:
        for blk in mb:
            # Apply viscous boundary conditions
            for face in blk.faces:
                face.bcFunc(blk, face, mb.eos, mb.thtrdat, "preDqDxyz", mb.titme)

            # Update spatial derivatives
            mb.dqdxyz(blk)

        # communicate derivatives
        communicate(mb, ["dqdx", "dqdy", "dqdz"])
        for blk in mb:
            # Apply spatial derivative boundary conditions
            for face in blk.faces:
                face.bcFunc(blk, face, mb.eos, mb.thtrdat, "postDqDxyz", mb.titme)

            # Apply subgrid model (must be after dqdxyz)
            mb.sgs(blk)

            # Diffusive fluxes
            mb.diffFlux(blk)
            mb.applyDiffFlux(blk, -1.0)  # <-- -1.0 is arbitrary, see applyFlux.cpp

    for blk in mb:
        # Chemical source terms
        mb.expChem(
            blk,
            mb.thtrdat,
            nChemSubSteps=mb.config["thermochem"]["nChemSubSteps"],
            dt=mb.config["timeIntegration"]["dt"],
        )
