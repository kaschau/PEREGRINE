from .compute import utils

"""
The right hand side of a peregrine case. Starts from dQ/dt=0 and
progressively flux differences based on the RHS attribute methods
set for the multiBlock solver.
"""


def RHS(mb):
    for blk in mb:
        # Zero out dQ array
        utils.dQzero(blk.cpp)

        # Primary advective fluxes
        mb.primaryAdvFlux(blk.cpp)
        mb.applyPrimaryAdvFlux(blk.cpp, 1.0)  # <-- 1.0 is for primary flux

        # Secondary advective fluxes
        mb.secondaryAdvFlux(blk.cpp)
        mb.applySecondaryAdvFlux(blk.cpp, 0.0)  # <-- 0.0 is for secondary flux

    if mb.config["RHS"]["diffusion"]:
        for blk in mb:
            # Apply viscous boundary conditions
            for face in blk.faces:
                face.bcFunc(
                    blk.cpp, face.cpp, mb.eos, mb.thtrdat.cpp, "preDqDxyz", mb.titme
                )

            # Update spatial derivatives
            mb.dqdxyz(blk.cpp)

        # communicate derivatives
        mb.communicator.exchange("grads")
        for blk in mb:
            # Apply spatial derivative boundary conditions
            for face in blk.faces:
                face.bcFunc(
                    blk.cpp, face.cpp, mb.eos, mb.thtrdat.cpp, "postDqDxyz", mb.titme
                )

            # Apply subgrid model (must be after dqdxyz)
            mb.sgs(blk.cpp)

            # Diffusive fluxes
            mb.diffFlux(blk.cpp)
            mb.applyDiffFlux(blk.cpp, -1.0)  # <-- -1.0 is arbitrary, see applyFlux.cpp

    for blk in mb:
        # Chemical source terms
        mb.expChem(
            blk.cpp,
            mb.thtrdat.cpp,
            nChemSubSteps=mb.config["mcPhysics"]["nChemSubSteps"],
            dt=mb.config["timeIntegration"]["dt"],
        )
