from .bcs import applyBcs
from .mpicomm.blockComm import communicate


def consistify(mb):

    # We assume that the interior of the blocks have a
    # conservative Q variable field. We update the
    # interior primatives, apply boundary conditions,
    # update halo values as needed, then communicate
    # everything.

    # First communicate conservatives
    communicate(mb, ["Q"])

    # Keep a list of arrays we need to communicate as we change them
    commList = []

    # Now update derived arrays for ENTIRE block,
    #  even exterior halos.
    for blk in mb:
        mb.eos(blk, mb.thtrdat, -1, "cons")

    # Apply euler boundary conditions
    applyBcs(mb, "euler")

    # Update transport properties
    for blk in mb:
        mb.trans(blk, mb.thtrdat, -1)

    # Update spatial derivatives
    for blk in mb:
        mb.dqdxyz(blk)

    # TODO: can we get rid of this if check?
    if mb.config["RHS"]["diffusion"]:
        # Apply viscous boundary conditions
        applyBcs(mb, "viscous")

        # communicate viscous halos
        commList += ["dqdx", "dqdy", "dqdz"]

    # Update switch
    for blk in mb:
        mb.switch(blk)
    if mb.switch.__name__ != 'null':
        commList += ["phi"]

    # Communicate necessary halos
    communicate(mb, commList)
