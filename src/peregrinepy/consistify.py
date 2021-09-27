from .bcs import apply_bcs
from .mpicomm.blockcomm import communicate
from .compute.transport import kinetic_theory
from .compute import flux


def consistify(mb):

    # We assume that the interior of the blocks have a
    # conservative Q variable field. We update the
    # interior primatives, apply boundary conditions,
    # update halo values as needed, then communicate
    # everything

    # First communicate conservatives
    communicate(mb, ["Q"])

    # Now update derived arrays for ENTIRE block, even exterior halos
    for blk in mb:
        mb.eos(blk, mb.thtrdat, -1, "cons")

    # Apply euler boundary conditions
    apply_bcs(mb, "euler")

    if mb.config["RHS"]["diffusion"]:

        # Update transport properties
        for blk in mb:
            kinetic_theory(blk, mb.thtrdat, -1)

        # Update spatial derivatives
        for blk in mb:
            flux.dqdxyz(blk)

        # Apply viscous boundary conditions
        apply_bcs(mb, "viscous")

        # communicate viscous halos
        communicate(mb, ["dqdx", "dqdy", "dqdz"])
