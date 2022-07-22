from .mpiComm import communicate


def consistify(mb):

    # We assume that the interior of the blocks have a
    # conservative Q variable field. We update the
    # interior primatives, apply boundary conditions,
    # update halo values as needed, then communicate
    # everything.

    # First communicate conservatives
    communicate(mb, ["Q"])

    # Now update derived arrays for ENTIRE block,
    #  even exterior halos.
    for blk in mb:
        mb.eos(blk, mb.thtrdat, -1, "cons")

        # Apply euler boundary conditions
        for face in blk.faces:
            face.bcFunc(blk, face, mb.eos, mb.thtrdat, "euler", mb.tme)

        if mb.config["RHS"]["diffusion"]:

            # Update transport properties
            mb.trans(blk, mb.thtrdat, -1)

            # Update spatial derivatives
            mb.dqdxyz(blk)

            # Apply viscous boundary conditions
            for face in blk.faces:
                face.bcFunc(blk, face, mb.eos, mb.thtrdat, "viscous", mb.tme)

            # Apply subgrid model
            mb.sgs(blk)

        # Update switch
        mb.switch(blk)

    # Communicate necessary halos
    communicate(mb, mb.commList)
