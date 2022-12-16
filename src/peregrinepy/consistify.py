from .mpiComm import communicate


def consistify(mb, given="cons"):

    # We assume that the interior of the blocks have a
    # conservative Q variable field. We update the
    # interior primatives, apply boundary conditions,
    # update halo values as needed, then communicate
    # everything.
    #
    # Since we are storing derivatives on faces now,
    # the calculation of spatial derivatives is moved
    # to RHS so viscous boundary conditions can be
    # applied after inviscid flux calculation, and
    # before viscous flux calculations.

    # First communicate conservatives
    if given == "cons":
        communicate(mb, ["Q"])
    elif given == "prims":
        communicate(mb, ["q"])

    # Now update derived arrays for ENTIRE block,
    #  even exterior halos.
    for blk in mb:
        mb.eos(blk, mb.thtrdat, -1, given)

        # Apply euler boundary conditions
        for face in blk.faces:
            face.bcFunc(blk, face, mb.eos, mb.thtrdat, "euler", mb.tme)

        # Update transport properties
        mb.trans(blk, mb.thtrdat, -1)

        # Apply subgrid model
        mb.sgs(blk)

        # Update switch
        mb.switch(blk)

        # Apply viscous sponge
        mb.viscousSponge(
            blk,
            mb.config["viscousSponge"]["origin"],
            mb.config["viscousSponge"]["ending"],
            mb.config["viscousSponge"]["multiplier"],
        )

    # Communicate necessary halos
    communicate(mb, mb.commList)
